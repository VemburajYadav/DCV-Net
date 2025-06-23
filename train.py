from __future__ import print_function, division
import sys
import argparse, configparser
import os
from glob import glob
import os.path as osp
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core.raft import RAFT
from core.dynamic_raft import DynamicRAFT
from att_raft import AttRAFT
from core.matching_loss import compute_supervision_coarse, backward_warp
import evaluate
from core import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    # Flow loss
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        i_loss = (valid[:, None] * i_loss).mean()
        flow_loss += i_weight * i_loss

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        "loss": i_loss.item()
    }

    return flow_loss, metrics


def sequence_loss_with_coarse_supervision(flow_preds, SoftCorrMap_list, image1, image2,
                                          flow_gt, valid,
                                          gamma=0.8, coarse_weight=0.01, coarse_gamma=0.8,
                                          coarse_loss_type="focal", focal_alpha=0.25, focal_gamma=2.0,
                                          max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    n_corr_maps = len(SoftCorrMap_list)
    flow_loss = 0.0
    matching_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    # Flow loss
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        i_loss = (valid[:, None] * i_loss).mean()
        flow_loss += i_weight * i_loss

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    # Matching loss
    img_2back1 = backward_warp(image2, flow_gt)
    occlusionMap = (image1 - img_2back1).mean(1, keepdims=True)  # (N, H, W)
    occlusionMap = torch.abs(occlusionMap) > 20
    occlusionMap = occlusionMap.float()

    conf_matrix_gt = compute_supervision_coarse(flow_gt, occlusionMap, 8)  # 8 from RAFT downsample

    c_pos_w = c_neg_w = 1.0
    pos_mask, neg_mask = conf_matrix_gt == 1, conf_matrix_gt == 0

    for i in range(n_corr_maps):
        i_match_weight = coarse_gamma ** (n_corr_maps - i - 1)
        i_conf = SoftCorrMap_list[i]
        i_conf = torch.clamp(i_conf, 1e-6, 1 - 1e-6)

        i_conf_pos, i_conf_neg = i_conf[pos_mask], i_conf[neg_mask]
        i_loss_pos = torch.log(i_conf_pos)
        i_loss_neg = torch.log(1 - i_conf_neg)

        if coarse_loss_type == "cross_entropy":
            i_match_loss = -(c_pos_w * i_loss_pos.mean() + c_neg_w * i_loss_neg.mean())
        elif coarse_loss_type == "focal":
            i_loss_pos_focal = -focal_alpha * torch.pow(1 - i_conf_pos, gamma) * i_loss_pos
            i_loss_neg_focal = -focal_alpha * torch.pow(i_conf_neg, gamma) * i_loss_neg
            i_match_loss = c_pos_w * i_loss_pos_focal.mean() + c_neg_w * i_loss_neg_focal.mean()
        else:
            raise NotImplementedError

        matching_loss += i_match_weight * i_match_loss

    total_loss = flow_loss + coarse_weight * matching_loss

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        "match_loss": i_match_loss.item(),
        "match_loss_pos": -i_loss_pos.mean().item(),
        "match_loss_neg": -i_loss_neg.mean().item(),
        "flow_loss": i_loss.item()
    }

    return total_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    if args.scheduler == "ConstantLR":
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.num_steps)
    elif args.scheduler == "MultiplicativeLR":
        lmbda = lambda epoch: 0.95
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif args.scheduler == "LinearLR":
        num_steps = args.num_steps - args.total_steps
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, num_steps + 100,
                                                  pct_start=0.0, cycle_momentum=False,
                                                  anneal_strategy='linear')
    else:
        pct_start = args.warmup_steps / args.num_steps
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
                                                  pct_start=pct_start, cycle_momentum=False,
                                                  anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, log_dir, total_steps=0):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        self.log_dir = log_dir
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    if args.att_raft:
        model = nn.DataParallel(AttRAFT(args), device_ids=args.gpus)
    else:
        if args.dynamic_matching:
            model = nn.DataParallel(DynamicRAFT(args), device_ids=args.gpus)
        else:
            model = nn.DataParallel(RAFT(args), device_ids=args.gpus)

    print("Parameter Count: %d" % count_parameters(model))

    model.cuda()
    total_steps = 0

    if args.restore_ckpt is not None:
        if args.resume_same:
            # Resume training for current experiment
            restore_ckpt_dir = os.path.join("experiments", args.restore_ckpt, "ckpt")
            if args.restore_ckpt_step_idx == -1:
                ckpt_list = sorted(glob(osp.join(restore_ckpt_dir, '*.pth')))
                restore_ckpt_path = ckpt_list[-1]
            else:
                restore_ckpt_path = os.path.join(restore_ckpt_dir,
                                                 '{:06d}_raft.pth'.format(args.restore_ckpt_step_idx))

            saved_state = torch.load(restore_ckpt_path)

            total_steps = saved_state["global_step"]
            args.total_steps = total_steps

            optimizer, scheduler = fetch_optimizer(args, model)
            optimizer.load_state_dict(saved_state["optimizer_state_dict"])

            if args.scheduler != "LinearLR":
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

            # if args.scheduler == "OneCycleLR":
            #     scheduler.load_state_dict(saved_state["scheduler_state_dict"])

            model.load_state_dict(saved_state["model_state_dict"], strict=False)
        else:
            optimizer, scheduler = fetch_optimizer(args, model)
            restore_ckpt_dir = os.path.join("experiments", args.restore_ckpt, "ckpt")
            if args.restore_ckpt_step_idx == -1:
                # ckpt_list = sorted(glob(osp.join(restore_ckpt_dir, '*.pth')))
                # restore_ckpt_path = ckpt_list[-1]
                restore_ckpt_path = os.path.join(restore_ckpt_dir, "raft.pth")
                model.load_state_dict(torch.load(restore_ckpt_path), strict=False)
            else:
                restore_ckpt_path = os.path.join(restore_ckpt_dir,
                                                 '{:06d}_raft.pth'.format(args.restore_ckpt_step_idx))
                # restore_ckpt_path = os.path.join("experiments", restore_ckpt_dir, "raft.pth")
                model.load_state_dict(torch.load(restore_ckpt_path)["model_state_dict"], strict=False)
            print("restore_ckpt_path", restore_ckpt_path)
    else:
        optimizer, scheduler = fetch_optimizer(args, model)

    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args.logs_dir, total_steps=total_steps)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            if args.coarse_supervision or args.dynamic_coarse_supervision:
                image1, image2, flow, valid, img1_no_aug, img2_no_aug = [x.cuda() for x in data_blob]
            else:
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            if args.coarse_supervision or args.dynamic_coarse_supervision:
                flow_predictions, SoftCorrMap_List = model(image1, image2, iters=args.iters)
                loss, metrics = sequence_loss_with_coarse_supervision(flow_predictions, SoftCorrMap_List,
                                                                      img1_no_aug, img2_no_aug, flow,
                                                                      valid, gamma=args.gamma,
                                                                      coarse_weight=args.coarse_loss_weight,
                                                                      coarse_gamma=args.coarse_loss_gamma,
                                                                      coarse_loss_type=args.coarse_loss_type,
                                                                      focal_alpha=args.coarse_loss_focal_alpha,
                                                                      focal_gamma=args.coarse_loss_focal_gamma)
            else:
                flow_predictions = model(image1, image2, iters=args.iters)
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)

            if args.scheduler == "MultiplicativeLR":
                if total_steps % 1000 == 0:
                    scheduler.step()
            else:
                scheduler.step()

            scaler.update()
            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = os.path.join(args.ckpt_dir, '{:06d}_raft.pth'.format(total_steps + 1))
                model_state_dict = model.state_dict()

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))
                    elif val_dataset == 'kitti_tile':
                        results.update(evaluate.validate_kitti_tile(model.module))
                save_state_dict = {
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': total_steps + 1,
                }

                save_state_dict.update(results)
                torch.save(save_state_dict, PATH)
                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = os.path.join(args.ckpt_dir, 'raft.pth')
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--resume_same', action='store_true', help='resume training')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--restore_ckpt_step_idx', type=int, default=-1,
                        help="step index of the stored checkpoint")

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--scheduler', type=str, default="OneCycleLR",
                        help="Learning rate scheduler to use during training")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use memory efficient implementation for correlation')

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
    parser.add_argument('--att_layer_norm', type=str, default="pre",
                        help="pre or post normalization in attention block")
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
    parser.add_argument('--att_first_no_share', action='store_true',
                        help='Separate attention block fore the first iteration')
    parser.add_argument('--att_use_mlp', action='store_true',
                        help='use mlp layer after attention')
    parser.add_argument('--att_no_pos_enc', action='store_true',
                        help='No positional encoding')
    parser.add_argument('--att_activation', type=str, default="ReLU",
                        help="activation in MLP layer of attention")
    parser.add_argument('--swin_att_num_splits', type=int, default=2)
    parser.add_argument('--att_use_hidden', action='store_true')
    parser.add_argument('--gma', action='store_true',
                        help='gma type motion feature aggregation')
    parser.add_argument('--gma_att_heads', type=int, default=1)
    parser.add_argument('--gma_agg_heads', type=int, default=1)

    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    seed = 1234
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_dir = os.path.join("./experiments")
    stage_dir = os.path.join(experiment_dir, args.name)

    if not os.path.isdir(stage_dir):
        os.makedirs(stage_dir)

    ckpt_dir = os.path.join(stage_dir, "ckpt")
    logs_dir = os.path.join(stage_dir, "logs")

    print("experiment base directory:", stage_dir)
    print("checkpoints dir:", ckpt_dir)
    print("logs dir:", logs_dir)

    for dirs in [ckpt_dir, logs_dir]:
        if not os.path.isdir(dirs):
            os.makedirs(dirs)

    args.stage_dir = stage_dir
    args.ckpt_dir = ckpt_dir
    args.logs_dir = logs_dir

    print(args)
    train(args)