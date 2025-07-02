import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils.utils import coords_grid, bilinear_sampler
import numpy as np


def cost_volume_to_matching_confidence(cost_volume: Tensor) -> Tensor:
    matching_prob_12 = F.softmax(cost_volume, dim=-1)
    matching_prob_21 = F.softmax(cost_volume, dim=1)
    matching_confidence = matching_prob_12 * matching_prob_21
    return matching_confidence


def backward_warp(img: Tensor, flow: Tensor) -> Tensor:
    b, _, h, w = img.size()

    coords_0 = coords_grid(b, h, w, img.device)
    coords_1 = coords_0 + flow
    coords_1 = coords_1.permute(0, 2, 3, 1)
    img_warped = bilinear_sampler(img, coords_1)

    return img_warped


@torch.no_grad()
def compute_supervision_coarse(flow, occlusions, scale: int, return_idx_only=False):
    N, _, H, W = flow.shape
    Hc, Wc = int(np.ceil(H / scale)), int(np.ceil(W / scale))

    occlusions_c = occlusions[:, :, ::scale, ::scale]
    flow_c = flow[:, :, ::scale, ::scale] / scale
    occlusions_c = occlusions_c.reshape(N, Hc * Wc)

    grid_c = coords_grid(N, Hc, Wc, flow_c.device)
    warp_c = grid_c + flow_c
    warp_c = warp_c.permute(0, 2, 3, 1).reshape(N, Hc * Wc, 2)
    warp_c = warp_c.round().long()

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    occlusions_c[out_bound_mask(warp_c, Wc, Hc)] = 1
    warp_c = warp_c[..., 0] + warp_c[..., 1] * Wc

    b_ids, i_ids = torch.split(torch.nonzero(occlusions_c == 0), 1, dim=1)
    j_ids = warp_c[b_ids, i_ids]

    if return_idx_only:
        return b_ids, i_ids, j_ids
    else:
        conf_matrix_gt = torch.zeros(N, Hc * Wc, Hc * Wc, device=flow.device)
        conf_matrix_gt[b_ids, i_ids, j_ids] = 1
        return conf_matrix_gt


def get_occlusion_map(img1, img2, flow, threshold=20):
    img1_warped = backward_warp(img2, flow)
    warp_diff = (img1 - img1_warped).abs().mean(1, keepdim=True)
    occlusion = (warp_diff > threshold).float()
    return occlusion



