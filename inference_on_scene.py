"""
This script evaluates optical flow models (RAFT variants) on image pairs, optionally saving visualized results.

Supports tiled and non-tiled inference with different datasets including TartanAir, Sintel, Spring, Kubrik, etc.
"""

# --- Standard Libraries ---
import sys
import os
import time
import math
import random
import argparse
import os.path as osp
from pathlib import Path
from glob import glob
import copy

# --- Image and Array Libraries ---
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2

# --- Project-Specific Imports ---
from core import datasets
from core.raft import RAFT
from core.att_raft import AttRAFT
from core.dynamic_raft import DynamicRAFT
from core.utils import flow_viz, frame_utils
from core.utils.frame_utils import read_gen
from core.utils.flow_io import readFlowFile, readPngFlowKubrik
from core.utils.utils import InputPadder, forward_interpolate
from core.utils.cv2_viz_utils import viz_img_grid, save_frame

TRAIN_SIZE = [432, 960]  # Default patch size for tiled inference


class InputPadder:
    """Pads images so their dimensions are divisible by 8, as required by RAFT"""

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8

        # Padding strategy varies based on dataset
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        elif mode == 'kitti432':
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == 'kitti376':
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        """Applies padding to input tensors"""
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self, x):
        """Removes previously added padding"""
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    """Computes grid coordinates for overlapping patches over the input image"""
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("Minimum overlap must be smaller than patch size.")

    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))

    # Ensure final patches align with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]

    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    """Computes Gaussian weights for blending tiled flow predictions"""
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
        weights[:, idx, h:h + patch_size[0], w:w + patch_size[1]] = weights_hw
    weights = weights.cuda()

    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx + 1, h:h + patch_size[0], w:w + patch_size[1]])

    return patch_weights


def inference_one_pair_tiled(model, img1, img2, sigma=0.05, iters=32):
    """
    Performs tiled inference on large images by splitting into overlapping patches.
    Aggregates flow using Gaussian-weighted averaging.
    """
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    image1 = img1[None].cuda()
    image2 = img2[None].cuda()
    IMAGE_SIZE = image1.shape[2:]

    # Adjust patch size if image is smaller
    if IMAGE_SIZE[1] < TRAIN_SIZE[1]:
        TRAIN_SIZE[1] = IMAGE_SIZE[1]
    if IMAGE_SIZE[0] < TRAIN_SIZE[0]:
        TRAIN_SIZE[0] = IMAGE_SIZE[0]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    flows = 0
    flow_count = 0

    for idx, (h, w) in enumerate(hws):
        image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
        image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]

        flow_low, flow_pr = model(image1_tile, image2_tile, iters=iters, test_mode=True)

        # Pad tiled flow to original image coordinates
        padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
        flows += F.pad(flow_pr * weights[idx], padding)
        flow_count += F.pad(weights[idx], padding)

    flow_pre = flows / flow_count
    flow_pre = flow_pre[0].cpu().permute(1, 2, 0).numpy()
    return flow_pre


def inference_one_pair(model, img1, img2, sigma=0.05, iters=32):
    """Performs standard (non-tiled) inference on one image pair"""
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    image1 = img1[None].cuda()
    image2 = img2[None].cuda()

    padder = InputPadder(img1.shape, mode="sintel")
    image1, image2 = padder.pad(image1, image2)

    flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
    flow_pre = padder.unpad(flow_pr[0]).cpu().permute(1, 2, 0).numpy()
    return flow_pre


def load_sample(img1_path, img2_path, flow_gt_path=None, invalid_mask_path=None, dataset=None):
    """
    Loads two consecutive images, and optionally the ground truth optical flow and invalid mask.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        flow_gt_path (str, optional): Path to ground truth optical flow.
        invalid_mask_path (str, optional): Path to the invalid flow mask.
        dataset (str, optional): Name of dataset (to apply custom handling).

    Returns:
        Tuple: (image1, image2, flow_gt, invalid_mask)
            - image1 (np.ndarray): First input image.
            - image2 (np.ndarray): Second input image.
            - flow_gt (np.ndarray or None): Ground truth flow, if available.
            - invalid_mask (np.ndarray or None): Binary mask of invalid flow pixels.
    """
    print(img1_path, img2_path, flow_gt_path)
    image1 = np.array(read_gen(img1_path)).astype(np.uint8)
    image2 = np.array(read_gen(img2_path)).astype(np.uint8)

    # Remove alpha channel for 'kubrik-nk' dataset (only RGB channels)
    if dataset is not None and dataset == "kubrik-nk":
        image1 = image1[:, :, :3]
        image2 = image2[:, :, :3]

    if flow_gt_path is not None:
        # Dataset-specific flow loading
        if dataset is not None and dataset == "kubrik-nk":
            flow_gt = readPngFlowKubrik(flow_gt_path).astype(np.float32)
            flow_gt = flow_gt[:, :, [1, 0]]  # Swap flow channels (v, u)
        else:
            flow_gt = readFlowFile(flow_gt_path).astype(np.float32)

        # Downsample flow if required by dataset
        if dataset is not None and dataset == "spring":
            flow_gt = flow_gt[::2, ::2]

        # Load invalid flow mask or compute it from flow values
        if invalid_mask_path is not None:
            invalid_mask = readFlowFile(invalid_mask_path).astype(np.float32) > 0
        else:
            flow_u, flow_v = flow_gt[:, :, 0], flow_gt[:, :, 1]
            invalid_mask_nan = np.logical_or(np.isnan(flow_u), np.isnan(flow_v))
            invalid_mask_thresh = np.logical_or(np.abs(flow_u) > 10000, np.abs(flow_v) > 10000)
            invalid_mask = np.logical_or(invalid_mask_nan, invalid_mask_thresh)
            flow_gt[invalid_mask, :] = 0.0  # Zero-out invalid flow

        return image1, image2, flow_gt, invalid_mask

    return image1, image2, None, None


def get_input_paths(scene_dir, dataset='tartanair', load_gt_flow=True):
    """
    Returns a list of image pairs and (optionally) their corresponding flow and mask paths.

    Args:
        scene_dir (Path): Root path of the scene folder.
        dataset (str): Dataset name for dataset-specific handling.
        load_gt_flow (bool): Whether to load ground truth optical flow.

    Returns:
        Tuple: (img_pairs_list, flows_list, masks_list)
    """
    if dataset == "tartanair":
        # Tartanair dataset: load left camera images and npy-based flow/mask
        image_dir = scene_dir / "image_left"
        flow_dir = scene_dir / "flow"
        images_list = list(sorted(glob(os.path.join(str(image_dir), "*.png"))))

        if load_gt_flow:
            flows_list = list(sorted(glob(os.path.join(str(flow_dir), "*flow.npy"))))
            masks_list = list(sorted(glob(os.path.join(str(flow_dir), "*mask.npy"))))
        else:
            flows_list = [None for _ in range(len(images_list) - 1)]
            masks_list = [None for _ in range(len(images_list) - 1)]

        img_pairs_list = [[images_list[i], images_list[i + 1]] for i in range(len(images_list) - 1)]

    elif dataset == "spring":
        # Spring dataset: downsample flow files
        image_dir = scene_dir / "frame_left"
        flow_dir = scene_dir / "flow_FW_left"
        images_list = list(sorted(glob(os.path.join(str(image_dir), "*.png"))))

        if load_gt_flow:
            flows_list = list(sorted(glob(os.path.join(str(flow_dir), "*.flo5"))))[:-1]
        else:
            flows_list = [None for _ in range(len(images_list) - 1)]
        masks_list = [None for _ in range(len(images_list) - 1)]

        img_pairs_list = [[images_list[i], images_list[i + 1]] for i in range(len(images_list) - 1)]

    elif dataset == "kubrik-nk":
        # Kubrik NK dataset: construct flow directory path based on naming convention
        image_dir = copy.copy(scene_dir)
        scene = str(scene_dir).split("/")[-1]
        base_dir = str(scene_dir).split("/")[:-4]
        base_dir = "/".join(base_dir)
        flow_dir = Path(base_dir) / "forward_flow_1k" / "forward_flow" / "1k" / scene

        images_list = list(sorted(glob(os.path.join(str(image_dir), "*.png"))))

        if load_gt_flow:
            flows_list = list(sorted(glob(os.path.join(str(flow_dir), "*.png"))))[:-1]
        else:
            flows_list = [None for _ in range(len(images_list) - 1)]
        masks_list = [None for _ in range(len(images_list) - 1)]

        img_pairs_list = [[images_list[i], images_list[i + 1]] for i in range(len(images_list) - 1)]

    elif dataset == "sintel":
        # Sintel dataset: .flo format flow files
        image_dir = copy.copy(scene_dir)
        scene = str(scene_dir).split("/")[-1]
        base_dir = str(scene_dir).split("/")[:-2]
        base_dir = "/".join(base_dir)
        flow_dir = Path(base_dir) / "flow" / scene

        images_list = list(sorted(glob(os.path.join(str(image_dir), "*.png"))))

        if load_gt_flow:
            flows_list = list(sorted(glob(os.path.join(str(flow_dir), "*.flo"))))
        else:
            flows_list = [None for _ in range(len(images_list) - 1)]
        masks_list = [None for _ in range(len(images_list) - 1)]

        img_pairs_list = [[images_list[i], images_list[i + 1]] for i in range(len(images_list) - 1)]

    elif dataset == "bonn-dynamic":
        # Bonn Dynamic Scenes: images only
        image_dir = Path(scene_dir) / "rgb"
        images_list = list(sorted(glob(os.path.join(str(image_dir), "*.png"))))
        img_pairs_list = [[images_list[i], images_list[i + 1]] for i in range(len(images_list) - 1)]
        flows_list = [None for _ in range(len(images_list) - 1)]
        masks_list = [None for _ in range(len(images_list) - 1)]

    elif dataset == "h2o3d":
        # H2O3D dataset: sample every 5th frame
        image_dir = Path(scene_dir) / "rgb"
        images_list = list(sorted(glob(os.path.join(str(image_dir), "*.jpg"))))[::5]
        img_pairs_list = [[images_list[i], images_list[i + 1]] for i in range(len(images_list) - 1)]
        flows_list = [None for _ in range(len(images_list) - 1)]
        masks_list = [None for _ in range(len(images_list) - 1)]

    return img_pairs_list, flows_list, masks_list


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # Checkpoint loading
    parser.add_argument('--model', help="Path to restore model checkpoint from")
    parser.add_argument('--ckpt_step_idx', type=int, default=-1, help="Checkpoint index to load (-1 for latest)")

    # Model configuration
    parser.add_argument('--tile_arch', action='store_true', help='Enable tiled architecture inference')
    parser.add_argument('--name', default='raft-things', help="Name of the experiment")
    parser.add_argument('--stage', default='things', help="Training stage or variant (e.g., wo-tile)")
    parser.add_argument('--dataset', type=str, nargs='+', help="Dataset(s) for evaluation")
    parser.add_argument('--small', action='store_true', help='Use smaller model version')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='Use efficient correlation implementation')
    parser.add_argument('--encoder', type=str, default="cnn", help="Feature extraction backbone")
    parser.add_argument('--pretrain_imagenet', action='store_true', help='Use pretrained weights from ImageNet')

    # Cost volume similarity options
    parser.add_argument('--cosine_sim', action='store_true', help='Use cosine similarity in cost volume')
    parser.add_argument('--cosine_simv2', action='store_true', help='Cosine sim with feature map norm')
    parser.add_argument('--mixed_sim', action='store_true', help='Use both dot product and cosine similarity')
    parser.add_argument('--cv_softmax', action='store_true', help='Apply softmax on cost volume')
    parser.add_argument('--lookup_softmax', action='store_true', help='Apply softmax after cost volume lookup')
    parser.add_argument('--lookup_softmax_all', action='store_true', help='Apply softmax on all levels after lookup')

    # Dynamic matching and attention-based RAFT
    parser.add_argument('--dynamic_matching', action='store_true', help='Enable attention-based dynamic cost volume')
    parser.add_argument('--att_raft', action='store_true', help='Use GMA-style attention aggregation')
    parser.add_argument('--dynamic_motion_encoder', action='store_true', help='Use learnable motion encoder per block')

    # Coarse supervision configuration
    parser.add_argument('--coarse_supervision', action='store_true', help='Supervise using sparse cost volume matches')
    parser.add_argument('--dynamic_coarse_supervision', action='store_true', help='Sparse supervision on dynamic CV')
    parser.add_argument('--coarse_loss_weight', type=float, default=0.01, help='Weight of coarse loss')
    parser.add_argument('--coarse_loss_gamma', type=float, default=0.9, help='Gamma for exponential weighting')
    parser.add_argument('--coarse_loss_type', type=str, default="cross_entropy", help="Loss type: focal or cross_entropy")
    parser.add_argument('--coarse_loss_focal_gamma', type=float, default=2.0, help='Gamma for focal loss')
    parser.add_argument('--coarse_loss_focal_alpha', type=float, default=0.25, help='Alpha for focal loss')

    # Attention transformer configurations
    parser.add_argument('--att_nhead', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--att_layer_layout', type=str, nargs='+', default=["self", "cross"], help="Attention layer order")
    parser.add_argument('--att_layer_type', type=str, default="linear", help="Type of attention layer")
    parser.add_argument('--att_weight_share_after', type=int, default=-1, help='Weight sharing from iteration N onward')
    parser.add_argument('--att_fix_n_updates', action='store_true', help='Use fixed number of dynamic updates')
    parser.add_argument('--att_update_stride', type=int, default=1, help='Stride for dynamic update')
    parser.add_argument('--att_n_repeats', type=int, default=1, help='Repeats of transformer blocks')
    parser.add_argument('--att_share_qk_proj', action='store_true', help='Share query/key projection weights')
    parser.add_argument('--att_layer_norm', type=str, default="pre", help='Norm type: pre or post')
    parser.add_argument('--att_first_no_share', action='store_true', help='Use a separate block for first iteration')
    parser.add_argument('--att_use_mlp', action='store_true', help='Add MLP after attention block')
    parser.add_argument('--att_activation', type=str, default="ReLU", help='Activation function for MLP')
    parser.add_argument('--att_no_pos_enc', action='store_true', help='Disable positional encoding')
    parser.add_argument('--swin_att_num_splits', type=int, default=1, help='Swin-style split count')
    parser.add_argument('--att_use_hidden', action='store_true', help='Use hidden features in attention')

    # GMA configuration
    parser.add_argument('--gma', action='store_true', help='Enable GMA-style motion aggregation')
    parser.add_argument('--gma_att_heads', type=int, default=1, help='Attention heads for GMA')
    parser.add_argument('--gma_agg_heads', type=int, default=1, help='Aggregation heads for GMA')

    # RAFT and inference configuration
    parser.add_argument('--embedding_dim', type=int, default=256, help='Feature embedding dimensionality')
    parser.add_argument('--corr_radius', type=int, default=4, help='Cost volume correlation radius')
    parser.add_argument('--iter_sintel', type=int, default=32, help='Iterations for Sintel')
    parser.add_argument('--iter_kitti', type=int, default=24, help='Iterations for KITTI')
    parser.add_argument('--sintel_tile_sigma', type=float, default=0.05, help='Tile weighting (sigma)')

    # Paths for input/output
    parser.add_argument('--scene_dir', type=str, default="linear", help="Scene input directory path")
    parser.add_argument('--save_dir', type=str, default="linear", help="Directory to save results")
    parser.add_argument('--save_images', action='store_true', help='Save input image pairs')
    parser.add_argument('--save_flow_pred', action='store_true', help='Save predicted flow visualizations')
    parser.add_argument('--save_flow_gt', action='store_true', help='Save ground truth flow visualizations')
    parser.add_argument('--save_flow_pred_with_epe', action='store_true', help='Save flow prediction with EPE shown')

    # Set manual seed for full reproducibility across runs
    seed = 1234
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensures reproducible results at the cost of performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse command-line arguments
    args = parser.parse_args()

    print("Name of the experiment under evaluation: ", args.name)
    print(args)

    # Convert string paths to pathlib.Path for consistency
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.save_dir)

    # Initialize the model depending on the specified architecture
    if args.att_raft:
        model = torch.nn.DataParallel(AttRAFT(args, mode="eval"), device_ids=[0])
    else:
        if args.dynamic_matching:
            model = torch.nn.DataParallel(DynamicRAFT(args, mode='eval'), device_ids=[0])
        else:
            model = torch.nn.DataParallel(RAFT(args), device_ids=[0])

    # Load model weights from checkpoint
    if args.model is not None:
        # Load from user-specified model path
        ckpt_file = args.model
    else:
        # Load from default experiment directory
        experiment_dir = os.path.join("./experiments")
        stage_dir = os.path.join(experiment_dir, args.name)
        ckpt_dir = os.path.join(stage_dir, "ckpt")

        if args.ckpt_step_idx == -1:
            # Load latest/default checkpoint
            ckpt_file = os.path.join(ckpt_dir, 'raft.pth')
            model.load_state_dict(torch.load(ckpt_file))
        else:
            # Load checkpoint from specific training step
            ckpt_file = os.path.join(ckpt_dir, '{:06d}_raft.pth'.format(args.ckpt_step_idx))
            model.load_state_dict(torch.load(ckpt_file)['model_state_dict'])

    # Move model to GPU and set to evaluation mode
    model.cuda()
    model.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        print("dataset: ", args.dataset)

        # Get list of image pairs and corresponding GT flows and masks
        img_pairs_list, flows_list, masks_list = get_input_paths(scene_dir, dataset=args.dataset[0])

        for i in range(len(img_pairs_list)):
            # Load sample: image1, image2, ground-truth flow, and invalid mask (if any)
            image1, image2, flow_gt, invalid_mask = load_sample(
                img_pairs_list[i][0],
                img_pairs_list[i][1],
                flows_list[i],
                masks_list[i],
                dataset=args.dataset[0]
            )

            # Run inference depending on whether tiling is used
            if "wo-tile" in args.stage:
                flow_pred = inference_one_pair(model.module, image1, image2)
            else:
                flow_pred = inference_one_pair_tiled(model.module, image1, image2)

            # Compute flow magnitude for visualization
            if flow_gt is not None:
                flow_u, flow_v = flow_gt[:, :, 0], flow_gt[:, :, 1]
                if invalid_mask is not None:
                    valid_pixels_mask = np.logical_not(invalid_mask)
                    max_mag = np.sqrt(flow_u[valid_pixels_mask]**2 + flow_v[valid_pixels_mask]**2).max()
                else:
                    max_mag = np.sqrt(flow_u**2 + flow_v**2).max()
            else:
                max_mag = None

            # Save input images if enabled
            if args.save_images:
                img1_dir = output_dir / "image1"
                img2_dir = output_dir / "image2"

                img1_path = os.path.join(str(img1_dir), os.path.basename(img_pairs_list[i][0]))
                img2_path = os.path.join(str(img2_dir), os.path.basename(img_pairs_list[i][1]))

                save_frame(image1, img1_path)
                save_frame(image2, img2_path)

            # Save ground truth flow visualization if enabled
            if args.save_flow_gt:
                flow_gt_dir = output_dir / "flow_gt"
                flow_gt_path = os.path.join(
                    str(flow_gt_dir),
                    os.path.basename(flows_list[i]).split(".")[0] + ".png"
                )
                flow_gt_img = flow_viz.flow_to_image(flow_gt, max_scale=max_mag)
                save_frame(flow_gt_img, flow_gt_path)

            # Save predicted flow visualization if enabled
            if args.save_flow_pred:
                flow_pred_dir = output_dir / args.stage / "flow_pred"
                if flows_list[i] is not None:
                    flow_pred_path = os.path.join(
                        str(flow_pred_dir),
                        os.path.basename(flows_list[i]).split(".")[0] + ".png"
                    )
                else:
                    flow_pred_path = os.path.join(
                        str(flow_pred_dir),
                        os.path.basename(img_pairs_list[i][0])
                    )
                flow_pred_img = flow_viz.flow_to_image(flow_pred, max_scale=max_mag)
                save_frame(flow_pred_img, flow_pred_path)

            # Save predicted flow with EPE (End-Point Error) overlay if enabled
            if args.save_flow_pred_with_epe and flow_gt is not None:
                flow_pred_with_epe_dir = output_dir / args.stage / "flow_pred_with_epe"
                flow_pred_with_epe_path = os.path.join(
                    str(flow_pred_with_epe_dir),
                    os.path.basename(flows_list[i]).split(".")[0] + ".png"
                )

                # Compute EPE over valid pixels only
                valid_mask = np.logical_not(invalid_mask)
                epe = np.sqrt(np.sum((flow_gt - flow_pred) ** 2, axis=2))
                epe = epe.reshape(-1)[valid_mask.reshape(-1)].mean()

                # Overlay EPE on flow prediction
                epe_str = "EPE: {:.2f}".format(epe)
                str_grid = [[epe_str]]
                viz_frame = viz_img_grid([[flow_pred_img]], str_grid=str_grid, spacing=0)
                save_frame(viz_frame, flow_pred_with_epe_path, convert_to_bgr=False)













