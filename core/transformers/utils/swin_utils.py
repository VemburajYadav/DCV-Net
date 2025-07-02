import torch
import torch.nn as nn
import torch.nn.functional as F
from ...transformers.utils.position_encoding import PositionEncodingSine


def split_feature(feature,
                  num_splits=2,
                  channel_last=False,
                  ):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                               ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(splits,
                 num_splits=2,
                 channel_last=False,
                 ):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

    return merge


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std

    return img0, img1


class SwinPosEncoding(nn.Module):
    def __init__(self, attn_splits=2, feature_channels=256):
        super(SwinPosEncoding, self).__init__()

        self.attn_splits = attn_splits
        self.feature_channels = feature_channels

        self.pos_enc = PositionEncodingSine(d_model=feature_channels)

    def forward(self, feature0, feature1):

        attn_splits = self.attn_splits

        if attn_splits > 1:  # add position in splited window
            feature0_splits = split_feature(feature0, num_splits=attn_splits)
            feature1_splits = split_feature(feature1, num_splits=attn_splits)

            feature0_splits = self.pos_enc(feature0_splits)
            feature1_splits = self.pos_enc(feature1_splits)

            feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
            feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
        else:
            position = self.pos_enc(feature0)

            feature0 = feature0 + position
            feature1 = feature1 + position

        return feature0, feature1


class SwinShiftedWindowAttnMask(nn.Module):
    def __init__(self, input_resolution=(46, 62), attn_splits=2):
        super(SwinShiftedWindowAttnMask, self).__init__()

        window_size_h = input_resolution[0] // attn_splits
        window_size_w = input_resolution[1] // attn_splits

        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        h, w = input_resolution
        img_mask = torch.zeros((1, h, w, 1))  # 1 H W 1

        h_slices = (slice(0, -window_size_h),
                    slice(-window_size_h, -shift_size_h),
                    slice(-shift_size_h, None))
        w_slices = (slice(0, -window_size_w),
                    slice(-window_size_w, -shift_size_w),
                    slice(-shift_size_w, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = split_feature(img_mask, num_splits=attn_splits, channel_last=True)

        mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        self.register_buffer('attn_mask', attn_mask, persistent=False)

    def forward(self, x):
        return self.attn_mask




