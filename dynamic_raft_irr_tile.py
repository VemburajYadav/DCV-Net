import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.update import BasicUpdateBlock, SmallUpdateBlock, FlowEncoder, BasicUpdateBlockNoEncoder, BasicMotionEncoder
from core.extractor import BasicEncoder, SmallEncoder
from core.transformers.backbone import twins_svt_large, twins_svt_large_context
from core.corr import CorrBlock, CudaCorrBlock, CosineCorrBlock, MixedCorrBlock
from core.transformers.utils.position_encoding import PositionEncodingSine
from core.transformers.feature_transformer import FeatureTransformer, FeatureFlowTransformer
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


def compute_grid_indices(image_shape, patch_size, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights


class DynamicIrrTileRAFT(nn.Module):
    def __init__(self, args):
        super(DynamicIrrTileRAFT, self).__init__()
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
        self.dynamic_motion_encoder = self.args.dynamic_motion_encoder
        self.no_share_first_block = self.args.att_first_no_share

        # Positional encoding
        self.pos_enc = PositionEncodingSine(d_model=self.embedding_dim)

        # Transformer block
        transformer_config = {"d_model": self.embedding_dim, "nhead": args.att_nhead,
                              "layer_names": args.att_layer_layout,
                              "attention": args.att_layer_type,
                              "layer_norm": args.att_layer_norm,
                              "share_qk_proj": args.att_share_qk_proj,
                              "use_mlp": args.att_use_mlp,
                              "activation": args.att_activation}

        self.is_flow_guided_attn = "flow-feature" in args.att_layer_layout
        # Non-shared transformer blocks
        if self.is_flow_guided_attn:
            feature_update_modules = [FeatureFlowTransformer(config=transformer_config)
                                      for _ in range(self.n_non_shared_att_blocks)]
            # Shared attention block
            if not self.fixed_updates:
                att_shared_block = FeatureFlowTransformer(config=transformer_config)
                # Transformer blocks module list
                feature_update_modules.append(att_shared_block)
            self.feature_update_modules = nn.ModuleList(feature_update_modules)

            # Also a module to encode flow vectors to feature vectors of same length as visual features
            self.flow_encoder = FlowEncoder(hidden_dim=self.embedding_dim, output_dim=self.embedding_dim, hidden_k=3, output_k=1)

            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        else:
            feature_update_modules = [FeatureTransformer(config=transformer_config)
                                      for _ in range(self.n_non_shared_att_blocks)]

            # Prepend an attention block if first attention block is unique
            if self.no_share_first_block:
                first_att_block = FeatureTransformer(config=transformer_config)
                feature_update_modules = [first_att_block] + feature_update_modules

            if self.dynamic_motion_encoder:

                motion_enc_modules = [BasicMotionEncoder(args) for _ in range(self.n_non_shared_att_blocks + 1)]

                # Prepend a motion encoder if first attention block is unique
                if self.no_share_first_block:
                    first_motion_enc = BasicMotionEncoder(args)
                    motion_enc_modules = [first_motion_enc] + motion_enc_modules

                self.motion_enc_modules = nn.ModuleList(motion_enc_modules)
                self.update_block = BasicUpdateBlockNoEncoder(self.args, hidden_dim=hdim)
            else:
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

    def feature_attention(self, f1, f2, itr):
        b, c, h, w = f1.shape
        f1 = rearrange(f1, "b c h w -> b (h w) c")
        f2 = rearrange(f2, "b c h w -> b (h w) c")

        idx = self.get_att_block_idx_from_itr(itr)
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

    def forward(self, image1, image2, iters=12, train_size=(400, 720),
                image10=None, image20=None, flow_init=None, upsample=True,
                test_mode=False, return_all_iters=False):
        """ Estimate optical flow between pair of frames """

        train_size = (376, 720)
        # train_size = (400, 720)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image_size = image1.shape[2:]

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

        hws = compute_grid_indices(image_size, train_size)
        weights = compute_weight(hws, image_size, train_size, 0.05)

        train_size_down = (train_size[0] // 8, train_size[1] // 8)

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            h_s, w_s = h // 8, w // 8
            fmap1_tile = fmap1[:, :, h_s:h_s + train_size_down[0], w_s:w_s + train_size_down[1]]
            fmap2_tile = fmap2[:, :, h_s:h_s + train_size_down[0], w_s:w_s + train_size_down[1]]
            net_tile = net[:, :, h_s:h_s + train_size_down[0], w_s:w_s + train_size_down[1]]
            inp_tile = inp[:, :, h_s:h_s + train_size_down[0], w_s:w_s + train_size_down[1]]

            b, c, h_down, w_down = fmap1_tile.shape

            image1_tile = image1[:, :, h:h + train_size[0], w:w + train_size[1]]
            # image2_tile = image2[:, :, h:h + train_size[0], w:w + train_size[1]]

            if self.args.dynamic_coarse_supervision or self.args.coarse_supervision:
                corr_fn = self.get_corr_fn(fmap1_tile, fmap2_tile, global_matching=True)
                smax_12 = F.softmax(corr_fn.corrMap, dim=2)
                SoftCorrMap_list = [smax_12 * F.softmax(corr_fn.corrMap, dim=1)]

            coords0, coords1 = self.initialize_flow(image1_tile)

            if flow_init is not None:
                coords1 = coords1 + flow_init
            elif self.args.dynamic_coarse_supervision or self.args.coarse_supervision:
                # Regress the flow from softmax probabilities for flow initialization
                coords1 = torch.matmul(smax_12, coords1.reshape(b, 2, h_down * w_down).permute(0, 2, 1))
                coords1 = coords1.permute(0, 2, 1).reshape(b, 2, h_down, w_down)

            if self.args.encoder == "twins" and self.args.att_no_pos_enc:
                fmap1_tile = fmap1_tile
                fmap2_tile = fmap2_tile
            else:
                fmap1_tile = self.pos_enc(fmap1_tile)
                fmap2_tile = self.pos_enc(fmap2_tile)

            if self.is_flow_guided_attn:
                flow_pre = coords1 - coords0
                flow_pre_up = upflow8(flow_pre)

                occ = get_occlusion_map(image10, image20, flow_pre_up, threshold=self.args.occ_threshold)
                b_ids, i_ids, j_ids = compute_supervision_coarse(flow_pre_up, occ, 8, return_idx_only=True)

                flow_enc = self.flow_encoder(flow_pre)
                fmap1_tile, fmap2_tile = self.feature_flow_attention(fmap1_tile, fmap2_tile, flow_enc, )
            else:
                fmap1_tile, fmap2_tile = self.feature_attention(fmap1_tile, fmap2_tile, 0)

            corr_fn = self.get_corr_fn(fmap1_tile, fmap2_tile, global_matching=self.args.dynamic_coarse_supervision)
            if self.args.dynamic_coarse_supervision:
                SoftCorrMap_list.append(F.softmax(corr_fn.corrMap, dim=2) * F.softmax(corr_fn.corrMap, dim=1))

            flow_predictions = []
            for itr in range(iters):

                first_iter = itr == 0
                coords1 = coords1.detach()

                if first_iter:
                    corr = corr_fn(coords1)  # index correlation volume
                else:
                    if self.n_repeats > 1:
                        itr_s = itr - 1 if self.no_share_first_block else itr
                        use_prev_cv = itr_s % self.update_stride > 0 or \
                                      itr_s >= (self.n_non_shared_att_blocks * self.n_repeats * self.update_stride)
                    else:
                        use_prev_cv = itr % self.update_stride > 0 or \
                                      itr >= (self.n_non_shared_att_blocks * self.update_stride)
                    fixed_att_update_itr = self.fixed_updates and use_prev_cv

                    if fixed_att_update_itr:
                        corr = corr_fn(coords1)  # index correlation volume
                    else:
                        fmap1_tile, fmap2_tile = self.feature_attention(fmap1_tile, fmap2_tile, itr)
                        corr_fn = self.get_corr_fn(fmap1_tile, fmap2_tile, global_matching=self.args.dynamic_coarse_supervision)

                        if self.args.dynamic_coarse_supervision:
                            SoftCorrMap_list.append(F.softmax(corr_fn.corrMap, dim=2) * F.softmax(corr_fn.corrMap, dim=1))
                        corr = corr_fn(coords1)  # index correlation volume

                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    if self.dynamic_motion_encoder:
                        idx_from_itr = self.get_att_block_idx_from_itr(itr)
                        motion_feat = self.motion_enc_modules[idx_from_itr](flow, corr)
                        net_tile, up_mask, delta_flow = self.update_block(net_tile, inp_tile, motion_feat)
                    else:
                        net_tile, up_mask, delta_flow = self.update_block(net_tile, inp_tile, corr, flow)

                # F(t+1) = F(t) + \Delta(t), 9
                coords1 = coords1 + delta_flow

                # upsample predictions
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)

                flow_predictions.append(flow_up)

            padding = (w, image_size[1] - w - train_size[1], h, image_size[0] - h - train_size[0], 0, 0)
            flows += F.pad(flow_up * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count

        return coords1 - coords0, flow_pre

            # if test_mode:
            #     if return_all_iters:
            #         return flow_predictions
            #     else:
            #         return coords1 - coords0, flow_up

        # if self.args.dynamic_coarse_supervision or self.args.coarse_supervision:
        #     return flow_predictions, SoftCorrMap_list
        # else:
        #     return flow_predictions
