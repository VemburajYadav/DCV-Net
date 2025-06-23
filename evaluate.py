import sys
from PIL import Image
import argparse
import os
import time
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
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='val')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        s_less_10_list= []
        s_10_to_40_list = []
        s_greater_40_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            #
            # image1 = image1[:, :432, :]
            # image2 = image2[:, :432, :]
            # flow_gt = flow_gt[:, :432, :]

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            # flow = flow_pr[0].cpu()
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe = epe.view(-1)
            epe_list.append(epe.numpy())

            flow_gt_mag = torch.sum(flow_gt**2, dim=0).sqrt()
            flow_gt_mag = flow_gt_mag.view(-1)

            valid_mask_less_10 = (flow_gt_mag < 10)
            valid_mask_10_to_40 = (flow_gt_mag >= 10) * (flow_gt_mag <= 40)
            valid_mask_greater_40 = (flow_gt_mag > 40)

            s_less_10_list.append(epe[valid_mask_less_10].numpy())
            s_10_to_40_list.append(epe[valid_mask_10_to_40].numpy())
            s_greater_40_list.append(epe[valid_mask_greater_40].numpy())

        epe_all = np.concatenate(epe_list)
        epe_s_less_10_all = np.concatenate(s_less_10_list)
        epe_s_10_to_40_all = np.concatenate(s_10_to_40_list)
        epe_s_greater_40_all = np.concatenate(s_greater_40_list)

        epe = np.mean(epe_all)
        epe_s_less_10 = np.mean(epe_s_less_10_all)
        epe_s_10_to_40 = np.mean(epe_s_10_to_40_all)
        epe_s_greater_40 = np.mean(epe_s_greater_40_all)

        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation Sintel(%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Validation Sintel(%s) EPE (Displacement range split) s<10: %f, s10-40: %f, s>40: %f" %
              (dstype, epe_s_less_10, epe_s_10_to_40, epe_s_greater_40))

        results["Sintel " + dstype + '_epe'] = np.mean(epe_list)
        results["Sintel " + dstype + '_s0_10'] = epe_s_less_10
        results["Sintel " + dstype + '_s10_40'] = epe_s_10_to_40
        results["Sintel " + dstype + '_s40+'] = epe_s_greater_40

    return results


@torch.no_grad()
def validate_sintel_tile(model, sigma=0.05, iters=32):
    """ Peform validation using the Sintel (train) split """

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    results = {}
    for dstype in ['clean', "final"]:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)

        epe_list = []
        s_less_10_list= []
        s_10_to_40_list = []
        s_greater_40_list = []

        for val_id in range(len(val_dataset)):
            # if val_id % 50 == 0:
            #     print(val_id)

            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

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
            epe_list.append(epe.numpy())

            flow_gt_mag = torch.sum(flow_gt**2, dim=0).sqrt()
            flow_gt_mag = flow_gt_mag.view(-1)

            valid_mask_less_10 = (flow_gt_mag < 10)
            valid_mask_10_to_40 = (flow_gt_mag >= 10) * (flow_gt_mag <= 40)
            valid_mask_greater_40 = (flow_gt_mag > 40)

            s_less_10_list.append(epe[valid_mask_less_10].numpy())
            s_10_to_40_list.append(epe[valid_mask_10_to_40].numpy())
            s_greater_40_list.append(epe[valid_mask_greater_40].numpy())

        epe_all = np.concatenate(epe_list)
        epe_s_less_10_all = np.concatenate(s_less_10_list)
        epe_s_10_to_40_all = np.concatenate(s_10_to_40_list)
        epe_s_greater_40_all = np.concatenate(s_greater_40_list)

        epe = np.mean(epe_all)
        epe_s_less_10 = np.mean(epe_s_less_10_all)
        epe_s_10_to_40 = np.mean(epe_s_10_to_40_all)
        epe_s_greater_40 = np.mean(epe_s_greater_40_all)

        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation Sintel(%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Validation Sintel(%s) EPE (Displacement range split) s<10: %f, s10-40: %f, s>40: %f" %
              (dstype, epe_s_less_10, epe_s_10_to_40, epe_s_greater_40))

        results["Sintel " + dstype + '_epe'] = np.mean(epe_list)
        results["Sintel " + dstype + '_s0_10'] = epe_s_less_10
        results["Sintel " + dstype + '_s10_40'] = epe_s_10_to_40
        results["Sintel " + dstype + '_s40+'] = epe_s_greater_40

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_kitti_tile(model, sigma=0.05, iters=24):
    IMAGE_SIZE = [376, 1242]
    TRAIN_SIZE = [376, 960]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 376
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder(image1.shape, mode='kitti376')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            # flow_pre, flow_low = model(image1_tile, image2_tile)
            flow_low, flow_pr = model(image1_tile, image2_tile, iters=iters, test_mode=True)

            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pr * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).cpu()
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


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
        for dataset in args.dataset:
            if dataset == 'chairs':
                validate_chairs(model.module)

            elif dataset == 'sintel':
                validate_sintel(model.module, iters=args.iter_sintel)

            elif dataset == 'kitti':
                validate_kitti(model.module, iters=args.iter_kitti)

            elif dataset == 'sintel_tile':
                validate_sintel_tile(model.module, iters=args.iter_sintel, sigma=args.sintel_tile_sigma)

            elif dataset == 'kitti_tile':
                validate_kitti_tile(model.module, iters=args.iter_kitti)


