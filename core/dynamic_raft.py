import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock, FlowEncoder, BasicUpdateBlockNoEncoder, BasicMotionEncoder, MergeVisualHidden
from .extractor import BasicEncoder, SmallEncoder
from .transformers.backbone import twins_svt_large, twins_svt_large_context
from .corr import CorrBlock, CudaCorrBlock, CosineCorrBlock, MixedCorrBlock
from .transformers.utils.position_encoding import PositionEncodingSine
from .transformers.feature_transformer import FeatureTransformer, FeatureFlowTransformer
from .transformers.utils.swin_utils import SwinPosEncoding, SwinShiftedWindowAttnMask
from .transformers.swin_attention import generate_shift_window_attn_mask
from .transformers.gma import GMAAttention, GMAAggregate
from .utils.utils import bilinear_sampler, coords_grid, upflow8
from .matching_loss import compute_supervision_coarse, get_occlusion_map
from einops import rearrange

# Fallback for AMP autocast if not supported in current PyTorch version
try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled): pass
        def __enter__(self): pass
        def __exit__(self, *args): pass


class DynamicRAFT(nn.Module):
    """
    DynamicRAFT: An optical flow estimation model built upon RAFT architecture with attention and dynamic cost volumes.
    """

    def __init__(self, args, mode='train'):
        super(DynamicRAFT, self).__init__()
        self.args = args

        # Initialize dimensions and parameters based on whether model is small or full
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

        # Set default values for optional arguments
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # Feature and context networks
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

        # Configurations for attention and cost volume update
        self.fixed_updates = self.args.att_fix_n_updates
        self.update_stride = self.args.att_update_stride
        self.n_non_shared_att_blocks = args.att_weight_share_after
        self.n_repeats = args.att_n_repeats
        self.dynamic_motion_encoder = self.args.dynamic_motion_encoder
        self.no_share_first_block = self.args.att_first_no_share
        self.attention_type = self.args.att_layer_type
        self.swin_attn_num_splits = args.swin_att_num_splits

        # GMA-specific configs
        self.is_gma = args.gma
        self.gma_att_heads = args.gma_att_heads
        self.gma_agg_heads = args.gma_agg_heads

        # Whether to merge visual and hidden features before attention
        self.merge_vis_hidden = args.att_use_hidden

        # Initialize positional encoding
        if self.attention_type == 'swin':
            self.pos_enc = SwinPosEncoding(attn_splits=self.swin_attn_num_splits, feature_channels=self.embedding_dim)
            if mode == 'train':
                self.sw_attn_mask_module = SwinShiftedWindowAttnMask(
                    input_resolution=(args.image_size[0] // 8, args.image_size[1] // 8),
                    attn_splits=self.swin_attn_num_splits
                )
        else:
            self.pos_enc = PositionEncodingSine(d_model=self.embedding_dim)

        # Transformer configuration dictionary
        transformer_config = {
            "d_model": self.embedding_dim,
            "nhead": args.att_nhead,
            "layer_names": args.att_layer_layout,
            "attention": args.att_layer_type,
            "layer_norm": args.att_layer_norm,
            "share_qk_proj": args.att_share_qk_proj,
            "use_mlp": args.att_use_mlp,
            "activation": args.att_activation,
            "swin_att_num_splits": args.swin_attn_num_splits
        }

        self.is_flow_guided_attn = "flow-feature" in args.att_layer_layout

        # Initialize attention modules
        if self.is_flow_guided_attn:
            feature_update_modules = [FeatureFlowTransformer(config=transformer_config)
                                      for _ in range(self.n_non_shared_att_blocks)]
            if not self.fixed_updates:
                att_shared_block = FeatureFlowTransformer(config=transformer_config)
                feature_update_modules.append(att_shared_block)
            self.feature_update_modules = nn.ModuleList(feature_update_modules)
            self.flow_encoder = FlowEncoder(hidden_dim=self.embedding_dim, output_dim=self.embedding_dim, hidden_k=3, output_k=1)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        else:
            feature_update_modules = [FeatureTransformer(config=transformer_config)
                                      for _ in range(self.n_non_shared_att_blocks)]
            if self.no_share_first_block:
                first_att_block = FeatureTransformer(config=transformer_config)
                feature_update_modules = [first_att_block] + feature_update_modules

            if self.dynamic_motion_encoder:
                motion_enc_modules = [BasicMotionEncoder(args) for _ in range(self.n_non_shared_att_blocks + 1)]
                if self.no_share_first_block:
                    first_motion_enc = BasicMotionEncoder(args)
                    motion_enc_modules = [first_motion_enc] + motion_enc_modules
                self.motion_enc_modules = nn.ModuleList(motion_enc_modules)
                self.update_block = BasicUpdateBlockNoEncoder(self.args, hidden_dim=hdim)
            else:
                self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

            if self.is_gma:
                self.gma_attn = GMAAttention(args=self.args, dim=cdim, heads=self.gma_att_heads, max_pos_size=160, dim_head=cdim)
                self.gma_agg = GMAAggregate(args=self.args, dim=128, dim_head=128, heads=self.args.gma_agg_heads)

            if self.merge_vis_hidden:
                self.vis_hidden = MergeVisualHidden(f_dim=self.embedding_dim, h_dim=128)

            if not self.fixed_updates:
                att_shared_block = FeatureTransformer(config=transformer_config)
                feature_update_modules.append(att_shared_block)
            self.feature_update_modules = nn.ModuleList(feature_update_modules)

    def freeze_bn(self):
        """ Freeze all batch normalization layers to evaluation mode """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Initialize coordinate grids to represent zero flow """
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow using convex combination weights """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def get_corr_fn(self, f1, f2, global_matching=False):
        """ Get correlation block depending on settings """
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
        """ Determine which transformer block to use at iteration `itr` """
        if self.n_repeats > 1:
            att_region = self.n_non_shared_att_blocks * self.update_stride * self.n_repeats
            if self.no_share_first_block:
                if itr == 0:
                    idx = 0
                else:
                    itr_s = itr - 1
                    idx = (itr_s // self.update_stride) % self.n_non_shared_att_blocks if itr_s < att_region else self.n_non_shared_att_blocks
                    idx += 1
            else:
                idx = (itr // self.update_stride) % self.n_non_shared_att_blocks if itr < att_region else self.n_non_shared_att_blocks
        else:
            idx = itr // self.update_stride if itr < self.n_non_shared_att_blocks * self.update_stride else self.n_non_shared_att_blocks
        return idx

    def feature_attention(self, f1, f2, itr, attn_mask=None):
        """ Apply feature attention block """
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
        """ Feature attention guided by flow """
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
        """ Get occlusion map by warping image2 and comparing to image1 """
        img1_warped = bilinear_sampler(img2, flow.permute(0, 2, 3, 1))
        occlusionMap = (img1 - img1_warped).mean(1, keepdims=True)
        occlusionMap = torch.abs(occlusionMap) > 20
        occlusionMap = occlusionMap.float()
        return occlusionMap

    def forward(self, image1, image2, iters=12, image10=None, image20=None, flow_init=None, upsample=True,
                test_mode=False, return_all_iters=False):
        """
        Forward pass of DynamicRAFT for estimating optical flow between image1 and image2.

        Args:
            image1, image2: Input image pairs (B x 3 x H x W)
            iters: Number of recurrent refinement iterations
            image10, image20: Optional additional inputs used for occlusion handling
            flow_init: Optional flow initialization
            upsample: Whether to upsample final flow
            test_mode: Enables inference-specific behavior
            return_all_iters: Whether to return all intermediate predictions

        Returns:
            If test_mode=True:
                final flow or all intermediate flows depending on return_all_iters
            If training:
                list of flow predictions (and optionally soft cost volume maps)
        """

        # Normalize inputs to [-1, 1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # Extract feature maps using the feature encoder
        with autocast(enabled=self.args.mixed_precision):
            if self.args.encoder == "twins":
                fmap1 = self.fnet(image1)
                fmap2 = self.fnet(image2)
            else:
                fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        b, c, h, w = fmap1.shape

        # Extract context features and split into hidden state (net) and input (inp)
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            if self.is_gma:
                gma_attn = self.gma_attn(inp)

        # If performing coarse supervision, compute initial correlation map
        if self.args.dynamic_coarse_supervision or self.args.coarse_supervision:
            corr_fn = self.get_corr_fn(fmap1, fmap2, global_matching=True)
            smax_12 = F.softmax(corr_fn.corrMap, dim=2)
            SoftCorrMap_list = [smax_12 * F.softmax(corr_fn.corrMap, dim=1)]

        # Initialize flow coordinate grids
        coords0, coords1 = self.initialize_flow(image1)

        # Flow initialization logic
        if flow_init is not None:
            coords1 = coords1 + flow_init
        elif self.args.dynamic_coarse_supervision or self.args.coarse_supervision:
            coords1 = torch.matmul(smax_12, coords1.reshape(b, 2, h * w).permute(0, 2, 1))
            coords1 = coords1.permute(0, 2, 1).reshape(b, 2, h, w)

        # Apply positional encoding if needed
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

        # Optional flow-guided feature attention
        if self.is_flow_guided_attn:
            flow_pre = coords1 - coords0
            flow_pre_up = upflow8(flow_pre)

            # Generate occlusion map and matching indices
            occ = get_occlusion_map(image10, image20, flow_pre_up, threshold=self.args.occ_threshold)
            b_ids, i_ids, j_ids = compute_supervision_coarse(flow_pre_up, occ, 8, return_idx_only=True)

            flow_enc = self.flow_encoder(flow_pre)
            fmap1, fmap2 = self.feature_flow_attention(fmap1, fmap2, flow_enc, )
        else:
            fmap1, fmap2 = self.feature_attention(fmap1, fmap2, 0, attn_mask=attn_mask)

        # Initialize correlation function (may be updated during refinement)
        corr_fn = self.get_corr_fn(fmap1, fmap2, global_matching=self.args.dynamic_coarse_supervision)
        if self.args.dynamic_coarse_supervision:
            SoftCorrMap_list.append(F.softmax(corr_fn.corrMap, dim=2) * F.softmax(corr_fn.corrMap, dim=1))

        flow_predictions = []

        # Recurrent refinement loop
        for itr in range(iters):
            first_iter = itr == 0
            coords1 = coords1.detach()

            # Compute correlation only once or if required
            if first_iter:
                corr = corr_fn(coords1)
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
                    corr = corr_fn(coords1)
                else:
                    if self.merge_vis_hidden:
                        fmap1 = fmap1 + self.vis_hidden(fmap1, net)

                    fmap1, fmap2 = self.feature_attention(fmap1, fmap2, itr, attn_mask=attn_mask)
                    corr_fn = self.get_corr_fn(fmap1, fmap2, global_matching=self.args.dynamic_coarse_supervision)

                    if self.args.dynamic_coarse_supervision:
                        SoftCorrMap_list.append(F.softmax(corr_fn.corrMap, dim=2) * F.softmax(corr_fn.corrMap, dim=1))

                    corr = corr_fn(coords1)

            flow = coords1 - coords0

            # Update recurrent block using motion features or correlation directly
            with autocast(enabled=self.args.mixed_precision):
                if self.dynamic_motion_encoder:
                    idx_from_itr = self.get_att_block_idx_from_itr(itr)
                    motion_feat = self.motion_enc_modules[idx_from_itr](flow, corr)
                    if self.is_gma:
                        motion_feat = self.gma_agg(gma_attn, motion_feat)
                    net, up_mask, delta_flow = self.update_block(net, inp, motion_feat)
                else:
                    net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # Update flow estimate
            coords1 = coords1 + delta_flow

            # Upsample to full resolution
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        # Inference behavior
        if test_mode:
            if return_all_iters:
                return flow_predictions
            else:
                return coords1 - coords0, flow_up

        # Return flow predictions and optionally soft correlation maps
        if self.args.dynamic_coarse_supervision or self.args.coarse_supervision:
            return flow_predictions, SoftCorrMap_list
        else:
            return flow_predictions