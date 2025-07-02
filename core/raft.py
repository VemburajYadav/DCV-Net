import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .transformers.backbone import twins_svt_large, twins_svt_large_context
from .corr import CorrBlock, CudaCorrBlock, CosineCorrBlock, MixedCorrBlock
from .transformers.utils.position_encoding import PositionEncodingSine
from .transformers.feature_transformer import FeatureTransformer
from .utils.utils import bilinear_sampler, coords_grid, upflow8
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


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
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

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            if args.encoder == "cnn":
                self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
                self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            elif args.encoder == "twins":
                self.fnet = twins_svt_large(pretrained=args.pretrain_imagenet)
                self.cnet = twins_svt_large(pretrained=args.pretrain_imagenet)
            else:
                raise NotImplementedError

            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

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

        if self.args.alternate_corr:
            corr_fn = CudaCorrBlock(fmap1, fmap2, radius=self.args.corr_radius,
                                    lookup_softmax=self.args.lookup_softmax,
                                    lookup_softmax_all=self.args.lookup_softmax_all)
        else:
            if self.args.cosine_simv2:
                # Cosine similarity (Feature vectors are first normalized across the channel dimension
                # followed by a dot product between feature vectors of both the feature maps)
                corr_fn = CosineCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            elif self.args.mixed_sim:
                # Cosine similarity (Feature vectors of only the 1/8th level are first normalized across the
                # channel dimension followed by a dot product between feature vectors of both the feature maps).
                # Computed cost volume at 1/8th resolution is average pooled to obtain the down-sampled cost volumes.
                corr_fn = MixedCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            else:
                # cv_softmax: softmax across the dimensions of F2 (all the pixels)
                # lookup_softmax: Softmax across the dimensions of F2 after the lookup (for each level)
                # lookup_softmax_all: Softmax across the dimensions of F2 after the lookup (combined for all levels)
                corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius,
                                    cosine_sim=self.args.cosine_sim,
                                    cv_softmax=self.args.cv_softmax,
                                    lookup_softmax=self.args.lookup_softmax,
                                    lookup_softmax_all=self.args.lookup_softmax_all,
                                    global_matching=self.args.coarse_supervision)

        if self.args.coarse_supervision:
            smax_12 = F.softmax(corr_fn.corrMap, dim=2)
            SoftCorrMap_list = [smax_12 * F.softmax(corr_fn.corrMap, dim=1)]


        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        elif self.args.coarse_supervision:
            # Regress the flow from softmax probabilities for flow initialization
            coords1 = torch.matmul(smax_12, coords1.reshape(b, 2, h * w).permute(0, 2, 1))
            coords1 = coords1.permute(0, 2, 1).reshape(b, 2, h, w)

        flow_predictions = []
        for itr in range(iters):

            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
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

        if self.args.coarse_supervision:
            return flow_predictions, SoftCorrMap_list
        else:
            return flow_predictions
