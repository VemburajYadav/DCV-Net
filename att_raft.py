import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import BasicUpdateBlock, SmallUpdateBlock, FlowEncoder, BasicUpdateBlockNoEncoder, BasicMotionEncoder, MergeVisualHidden
from core.extractor import BasicEncoder, SmallEncoder
from core.transformers.backbone import twins_svt_large, twins_svt_large_context
from core.corr import CorrBlock, CudaCorrBlock, CosineCorrBlock, MixedCorrBlock
from core.transformers.utils.position_encoding import PositionEncodingSine
from core.transformers.feature_transformer import FeatureTransformer, FeatureFlowTransformer
from core.transformers.utils.swin_utils import SwinPosEncoding, SwinShiftedWindowAttnMask
from core.transformers.swin_attention import generate_shift_window_attn_mask
from core.transformers.gma import GMAAttention, GMAAggregate
from core.utils.utils import bilinear_sampler, coords_grid, upflow8
from core.matching_loss import compute_supervision_coarse, get_occlusion_map
from einops import rearrange

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class AttRAFT(nn.Module):
    def __init__(self, args, mode='train'):
        super(AttRAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            self.embedding_dim = args.embedding_dim

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            if args.encoder == "cnn":
                self.fnet = BasicEncoder(output_dim=self.embedding_dim, norm_fn='instance', dropout=args.dropout)
                self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            elif args.encoder == "twins":
                self.fnet = twins_svt_large(pretrained=args.pretrain_imagenet)
                self.cnet = twins_svt_large(pretrained=args.pretrain_imagenet)
            else:
                raise NotImplementedError

        # Modules for dynamic cost volume
        self.fixed_updates = self.args.att_fix_n_updates
        self.update_stride = self.args.att_update_stride
        self.n_non_shared_att_blocks = args.att_weight_share_after
        self.n_repeats = args.att_n_repeats
        self.no_share_first_block = self.args.att_first_no_share
        self.attention_type = self.args.att_layer_type
        self.swin_attn_num_splits = args.swin_att_num_splits

        # Positional encoding
        if self.attention_type == 'swin':
            self.pos_enc = SwinPosEncoding(attn_splits=self.swin_attn_num_splits, feature_channels=self.embedding_dim)

            if mode == 'train':
                self.sw_attn_mask_module = SwinShiftedWindowAttnMask(
                    input_resolution=(args.image_size[0] // 8, args.image_size[1] // 8),
                    attn_splits=self.swin_attn_num_splits
                )
        else:
            self.pos_enc = PositionEncodingSine(d_model=self.embedding_dim)

        # Transformer block
        transformer_config = {"d_model": self.embedding_dim, "nhead": args.att_nhead,
                              "layer_names": args.att_layer_layout,
                              "attention": args.att_layer_type,
                              "layer_norm": args.att_layer_norm,
                              "share_qk_proj": args.att_share_qk_proj,
                              "use_mlp": args.att_use_mlp,
                              "activation": args.att_activation,
                              "swin_att_num_splits": args.swin_att_num_splits}

        # Non-shared transformer blocks
        feature_update_modules = [FeatureTransformer(config=transformer_config)
                                  for _ in range(self.n_non_shared_att_blocks)]

        # Prepend an attention block if first attention block is unique
        if self.no_share_first_block:
            first_att_block = FeatureTransformer(config=transformer_config)
            feature_update_modules = [first_att_block] + feature_update_modules

        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        # Shared attention block
        if not self.fixed_updates:
            att_shared_block = FeatureTransformer(config=transformer_config)
            # Transformer blocks module list
            feature_update_modules.append(att_shared_block)
        self.feature_update_modules = nn.ModuleList(feature_update_modules)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def get_corr_fn(self, f1, f2, global_matching=False):
        if self.args.alternate_corr:
            corr_fn = CudaCorrBlock(f1, f2, radius=self.args.corr_radius,
                                    lookup_softmax=self.args.lookup_softmax,
                                    lookup_softmax_all=self.args.lookup_softmax_all)
        else:
            corr_fn = CorrBlock(f1, f2, radius=self.args.corr_radius,
                                cosine_sim=self.args.cosine_sim,
                                cv_softmax=self.args.cv_softmax,
                                lookup_softmax=self.args.lookup_softmax,
                                lookup_softmax_all=self.args.lookup_softmax_all,
                                global_matching=global_matching)

        return corr_fn

    def get_att_block_idx_from_itr(self, itr):
        if self.n_repeats > 1:
            att_region = self.n_non_shared_att_blocks * self.update_stride * self.n_repeats
            if self.no_share_first_block:
                if itr == 0:
                    idx = 0
                else:
                    itr_s = itr - 1
                    idx = (itr_s // self.update_stride) % self.n_non_shared_att_blocks if itr_s < att_region \
                        else self.n_non_shared_att_blocks
                    idx += 1
            else:
                idx = (itr // self.update_stride) % self.n_non_shared_att_blocks if itr < att_region \
                    else self.n_non_shared_att_blocks
        else:
            idx = itr // self.update_stride if itr < self.n_non_shared_att_blocks * self.update_stride \
                else self.n_non_shared_att_blocks
        return idx

    def feature_attention(self, f1, f2, itr, attn_mask=None):
        b, c, h, w = f1.shape
        f1 = rearrange(f1, "b c h w -> b (h w) c")
        f2 = rearrange(f2, "b c h w -> b (h w) c")

        with_shift = (itr % 2 == 1)
        idx = self.get_att_block_idx_from_itr(itr)
        if self.attention_type == "swin":
            f1, f2 = self.feature_update_modules[idx](f1, f2, h=h, w=w, attn_mask=attn_mask, with_shift=with_shift)
        else:
            f1, f2 = self.feature_update_modules[idx](f1, f2)

        f1 = rearrange(f1, "b (h w) c -> b c h w", h=h).contiguous()
        f2 = rearrange(f2, "b (h w) c -> b c h w", h=h).contiguous()

        return f1, f2

    def feature_flow_attention(self, f1, f2, flow_enc, match_idx0, match_idx1, itr):
        flow_with_pos_enc = self.pos_enc(flow_enc)

        b, c, h, w = f1.shape
        f1 = rearrange(f1, "b c h w -> b (h w) c")
        f2 = rearrange(f2, "b c h w -> b (h w) c")

        idx = self.get_att_block_idx_from_itr(itr)
        f1, f2 = self.feature_update_modules[idx](f1, f2)

        f1 = rearrange(f1, "b (h w) c -> b c h w", h=h).contiguous()
        f2 = rearrange(f2, "b (h w) c -> b c h w", h=h).contiguous()

        return f1, f2

    def get_sparse_matches_from_warping(self, img1, img2, flow):
        img1_warped = bilinear_sampler(img2, flow.permute(0, 2, 3, 1))
        occlusionMap = (img1 - img1_warped).mean(1, keepdims=True)  # (N, H, W)
        occlusionMap = torch.abs(occlusionMap) > 20
        occlusionMap = occlusionMap.float()

    def forward(self, image1, image2, iters=12, image10=None, image20=None, flow_init=None, upsample=True,
                test_mode=False, return_all_iters=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.encoder == "twins":
                fmap1 = self.fnet(image1)
                fmap2 = self.fnet(image2)
            else:
                fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        b, c, h, w = fmap1.shape

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        if self.args.encoder == "twins" and self.args.att_no_pos_enc:
            fmap1 = fmap1
            fmap2 = fmap2
        else:
            if self.attention_type == 'swin':
                fmap1, fmap2 = self.pos_enc(fmap1, fmap2)
                if test_mode:
                    window_size_h = h // self.swin_attn_num_splits
                    window_size_w = w // self.swin_attn_num_splits
                    shift_size_h = window_size_h // 2
                    shift_size_w = window_size_w // 2
                    attn_mask = generate_shift_window_attn_mask((h, w), window_size_h, window_size_w,
                                                                shift_size_h, shift_size_w, device=fmap1.device)
                else:
                    attn_mask = self.sw_attn_mask_module(fmap1)
            else:
                fmap1 = self.pos_enc(fmap1)
                fmap2 = self.pos_enc(fmap2)
                attn_mask = None

        if self.no_share_first_block:
            att_iters = 7
        else:
            att_iters = 6

        for att_itr in range(att_iters):
            fmap1, fmap2 = self.feature_attention(fmap1, fmap2, att_itr, attn_mask=attn_mask)

        corr_fn = self.get_corr_fn(fmap1, fmap2, global_matching=False)

        flow_predictions = []
        for itr in range(iters):

            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t), 9
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            if return_all_iters:
                return flow_predictions
            else:
                return coords1 - coords0, flow_up

        return flow_predictions
