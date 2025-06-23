import sys
from PIL import Image
import argparse
import os
import time
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from core import datasets
from core.utils import flow_viz
from core.utils import frame_utils
from glob import glob
import os.path as osp
from att_raft import AttRAFT
from core.raft import RAFT
from core.dynamic_raft import DynamicRAFT
from dynamic_raft_irr_tile import DynamicIrrTileRAFT
from core.utils.utils import InputPadder, forward_interpolate
from core.utils import flow_viz

TRAIN_SIZE = [432, 960]


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti432':
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == 'kitti376':
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
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


@torch.no_grad()
def validate_sintel_tile(model, sigma=0.05, iters=32, save_dir="/netscratch/vemburaj/predictions",
                        save_only_flow_preds=False, name="_dcv"):
    """ Peform validation using the Sintel (train) split """

    IMAGE_SIZE = [436, 1024]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    img1_dir = os.path.join(save_dir, "image1")
    img2_dir = os.path.join(save_dir, "image2")
    flow_gt_dir = os.path.join(save_dir, "flow_gt")
    flow_pred_dir = os.path.join(save_dir, "flow_pred" + name)

    if not save_only_flow_preds:
        if not os.path.isdir(img1_dir):
            os.makedirs(img1_dir)
        if not os.path.isdir(img2_dir):
            os.makedirs(img2_dir)
        if not os.path.isdir(flow_gt_dir):
            os.makedirs(flow_gt_dir)

    if not os.path.isdir(flow_pred_dir):
        os.makedirs(flow_pred_dir)

    model.eval()
    results = {}
    for dstype in ['clean']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)

        epe_list = []
        s_less_10_list= []
        s_10_to_40_list = []
        s_greater_40_list = []

        for val_id in range(len(val_dataset)):
            # if val_id % 50 == 0:
            #     print(val_id)

            image1_, image2_, flow_gt, _ = val_dataset[val_id]
            image1 = image1_[None].cuda()
            image2 = image2_[None].cuda()

            flows = 0
            flow_count = 0

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]

                # flow_pre, _ = model(image1_tile, image2_tile, flow_init=None)
                flow_low, flow_pr = model(image1_tile, image2_tile, iters=iters, test_mode=True)
                # flow = padder.unpad(flow_pr[0]).cpu()

                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pr * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow_pre = flows / flow_count
            flow_pre = flow_pre[0].cpu()

            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe = epe.view(-1)

            flow_gt_mag = torch.sum(flow_gt**2, dim=0).sqrt()
            flow_gt_mag = flow_gt_mag.view(-1)

            valid_mask_less_10 = (flow_gt_mag < 10)
            valid_mask_10_to_40 = (flow_gt_mag >= 10) * (flow_gt_mag <= 40)
            valid_mask_greater_40 = (flow_gt_mag > 40)

            epe_list.append(np.mean(epe.numpy()))
            s_less_10_list.append(np.mean(epe[valid_mask_less_10].numpy()))
            s_10_to_40_list.append(np.mean(epe[valid_mask_10_to_40].numpy()))
            s_greater_40_list.append(np.mean(epe[valid_mask_greater_40].numpy()))

            image1_ = image1_.permute(1, 2, 0).numpy().astype(np.uint8)
            image2_ = image2_.permute(1, 2, 0).numpy().astype(np.uint8)
            flow_p = flow_pre.permute(1, 2, 0).numpy()
            flow_gt_p = flow_gt.permute(1, 2, 0).numpy()

            # map flow to rgb image
            flow_vis = flow_viz.flow_to_image(flow_p)
            flow_gt_vis = flow_viz.flow_to_image(flow_gt_p)

            filename = "{:06d}.png".format(val_id)

            cv2.imwrite(os.path.join(flow_pred_dir, filename), flow_vis[:, :, [2, 1, 0]])

            if not save_only_flow_preds:
                cv2.imwrite(os.path.join(img1_dir, filename), image1_[:, :, [2, 1, 0]])
                cv2.imwrite(os.path.join(img2_dir, filename), image2_[:, :, [2, 1, 0]])
                cv2.imwrite(os.path.join(flow_gt_dir, filename), flow_gt_vis[:, :, [2, 1, 0]])

        epe_np = np.array(epe_list)
        s_10_np = np.array(s_less_10_list)
        s_10_to_40_np = np.array(s_10_to_40_list)
        s_40_np = np.array(s_greater_40_list)

        np.savez_compressed(os.path.join(save_dir, "metrics.npz"), epe=epe_np, s_10=s_10_np, s10_40=s_10_to_40_np,
                            s_40=s_40_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--ckpt_step_idx', type=int, default=-1)

    parser.add_argument('--tile_arch', action='store_true', help='tiled architecture')
    parser.add_argument('--name', default='raft-things', help="name your experiment")
    parser.add_argument('--dataset', type=str, nargs='+', help="dataset(s) for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--encoder', type=str, default="cnn",
                        help="Backbone for feature extraction")
    parser.add_argument('--pretrain_imagenet', action='store_true',
                        help='Load pretrained weights from imagenet')

    parser.add_argument('--cosine_sim', action='store_true', help='use cosine similarity for cost volume')
    parser.add_argument('--cosine_simv2', action='store_true', help='use cosine similarity for cost volume '
                                                                    '(feature map norm)')
    parser.add_argument('--mixed_sim', action='store_true', help='use dot product and cosine similarity for cost volume')
    parser.add_argument('--cv_softmax', action='store_true', help='apply softmax on the cost volume')
    parser.add_argument('--lookup_softmax', action='store_true', help='apply softmax on the cost volume after lookup')
    parser.add_argument('--lookup_softmax_all', action='store_true', help='apply softmax on the cost volume after '
                                                                          'lookup (all levels)')

    parser.add_argument('--dynamic_matching', action='store_true', help='Dynamic cost volume using attention on '
                                                                        'feature maps')
    parser.add_argument('--att_raft', action='store_true',
                        help='gma type motion feature aggregation')
    parser.add_argument('--dynamic_motion_encoder', action='store_true',
                        help='separate learnable motion encoder after each attention block')
    parser.add_argument('--coarse_supervision', action='store_true',
                        help='supervision with sparse matches on the cost volume')
    parser.add_argument('--dynamic_coarse_supervision', action='store_true',
                        help='supervision with sparse matches on the dynamic cost volume')
    parser.add_argument('--coarse_loss_weight', type=float, default=0.01, help='exponential weighting')
    parser.add_argument('--coarse_loss_gamma', type=float, default=0.9, help='exponential weighting')
    parser.add_argument('--coarse_loss_type', type=str, default="cross_entropy",
                        help="whether to use focal or cross entropy loss for supervision")
    parser.add_argument('--coarse_loss_focal_gamma', type=float, default=2.0,
                        help='alpha for focal loss')
    parser.add_argument('--coarse_loss_focal_alpha', type=float, default=0.25,
                        help='gamma for focal loss')

    parser.add_argument('--att_nhead', type=int, default=4, help="number of attention heads for multi-head attention")
    parser.add_argument('--att_layer_layout', type=str, nargs='+', default=["self","cross"],
                        help="layout of self and cross attention layers per feature update")
    parser.add_argument('--att_layer_type', type=str, default="linear",
                        help="type of attention to apply per attention layer. Options: linear, quadtree, full")
    parser.add_argument('--att_weight_share_after',  type=int, default=-1,
                        help='share the weights of the transformer block across the update '
                             'iterations except for the first '
                             'att_weight_share_after iterations')
    parser.add_argument('--att_fix_n_updates', action='store_true',
                        help='Fixed number of dynamic updates')
    parser.add_argument('--att_update_stride', type=int, default=1,
                        help="stride for dynamic cost volume")
    parser.add_argument('--att_n_repeats', type=int, default=1,
                        help="number of repeats for the transformer block")
    parser.add_argument('--att_share_qk_proj', action='store_true',
                        help='share same projection parameters for query and key of'
                             'visual feature attention')
    parser.add_argument('--att_layer_norm', type=str, default="pre",
                        help="pre or post normalization in attention block")
    parser.add_argument('--att_first_no_share', action='store_true',
                        help='Separate attention block fore the first iteration')
    parser.add_argument('--att_use_mlp', action='store_true',
                        help='use mlp layer after attention')
    parser.add_argument('--att_activation', type=str, default="ReLU",
                        help="activation in MLP layer of attention")
    parser.add_argument('--att_no_pos_enc', action='store_true',
                        help='No positional encoding')
    parser.add_argument('--swin_att_num_splits', type=int, default=1)
    parser.add_argument('--att_use_hidden', action='store_true')

    parser.add_argument('--gma', action='store_true',
                        help='gma type motion feature aggregation')
    parser.add_argument('--gma_att_heads', type=int, default=1)
    parser.add_argument('--gma_agg_heads', type=int, default=1)

    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--iter_sintel', type=int, default=32)
    parser.add_argument('--iter_kitti', type=int, default=24)
    parser.add_argument('--sintel_tile_sigma', type=float, default=0.05, help='exponential weighting')

    args = parser.parse_args()

    print("Name of the experiment under evaluation: ", args.name)
    print(args)

    if args.att_raft:
        model = torch.nn.DataParallel(AttRAFT(args, mode="eval"))
    else:
        if args.dynamic_matching:
            if args.tile_arch:
                model = torch.nn.DataParallel(DynamicIrrTileRAFT(args))
            else:
                model = torch.nn.DataParallel(DynamicRAFT(args, mode='eval'))
        else:
            model = torch.nn.DataParallel(RAFT(args))

    if args.model is not None:
        ckpt_file = args.model
    else:
        experiment_dir = os.path.join("./experiments")
        stage_dir = os.path.join(experiment_dir, args.name)
        ckpt_dir = os.path.join(stage_dir, "ckpt")
        if args.ckpt_step_idx == -1:
            ckpt_file = os.path.join(ckpt_dir, 'raft.pth')
            model.load_state_dict(torch.load(ckpt_file))
        else:
            ckpt_file = os.path.join(ckpt_dir, '{:06d}_raft.pth'.format(args.ckpt_step_idx))
            model.load_state_dict(torch.load(ckpt_file)['model_state_dict'])

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        validate_sintel_tile(model.module, iters=args.iter_sintel, sigma=args.sintel_tile_sigma, save_only_flow_preds=False)


