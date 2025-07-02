import os
import numpy as np
from glob import glob
import os.path as osp
import cv2

dcv_data = np.load(os.path.join("/home/vyadav/mnt/netscratch/predictions", "metrics.npz"))
ff_data = np.load(os.path.join("/home/vyadav/mnt/netscratch/predictions", "metrics_ff.npz"))

image1_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "image1")
image2_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "image2")
flow_gt_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "flow_gt")
flow_dcv_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "flow_pred_dcv")
flow_ff_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "flow_pred_ff")

image1_list = sorted(glob(osp.join(image1_dir, '*.png')))
image2_list = sorted(glob(osp.join(image2_dir, '*.png')))
flow_gt_list = sorted(glob(osp.join(flow_gt_dir, '*.png')))
flow_dcv_list = sorted(glob(osp.join(flow_dcv_dir, '*.png')))
flow_ff_list = sorted(glob(osp.join(flow_ff_dir, '*.png')))


best_epe_save_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "best_epe")
best_s10_save_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "best_s10")
best_s10_40_save_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "best_s10_40")
best_s40_save_dir = os.path.join("/home/vyadav/mnt/netscratch/predictions", "best_s40")

if not os.path.isdir(best_epe_save_dir):
    os.makedirs(best_epe_save_dir)

best_epe_image1_dir = os.path.join(best_epe_save_dir, "image1")
best_epe_image2_dir = os.path.join(best_epe_save_dir, "image2")
best_epe_flow_gt_dir = os.path.join(best_epe_save_dir, "flow_gt")
best_epe_flow_dcv_dir = os.path.join(best_epe_save_dir, "flow_dcv")
best_epe_flow_ff_dir = os.path.join(best_epe_save_dir, "flow_ff")

if not os.path.isdir(best_epe_image1_dir):
    os.makedirs(best_epe_image1_dir)
if not os.path.isdir(best_epe_image2_dir):
    os.makedirs(best_epe_image2_dir)
if not os.path.isdir(best_epe_flow_gt_dir):
    os.makedirs(best_epe_flow_gt_dir)
if not os.path.isdir(best_epe_flow_dcv_dir):
    os.makedirs(best_epe_flow_dcv_dir)
if not os.path.isdir(best_epe_flow_ff_dir):
    os.makedirs(best_epe_flow_ff_dir)

dcv_epe = dcv_data["epe"]
ff_epe = ff_data["epe"]

dcv_s10 = dcv_data["s_10"]
dcv_s10_40 = dcv_data["s10_40"]
dcv_s40 = dcv_data["s_40"]

ff_s10 = ff_data["s_10"]
ff_s10_40 = ff_data["s10_40"]
ff_s40 = ff_data["s_40"]

dcv_less_epe = np.where(dcv_epe <= ff_epe)[0]
dcv_less_s10 = np.where(dcv_s10 <= ff_s10)[0]
dcv_less_s10_40 = np.where(dcv_s10_40 <= ff_s10_40)[0]
dcv_less_s40 = np.where(dcv_s40 <= ff_s40)[0]

for id in range(len(dcv_less_epe)):
    idx = dcv_less_epe[id]
    img1 = cv2.imread(image1_list[idx])
    img2 = cv2.imread(image2_list[idx])
    flow_gt = cv2.imread(flow_gt_list[idx])
    flow_dcv = cv2.imread(flow_dcv_list[idx])
    flow_ff = cv2.imread(flow_ff_list[idx])

    cv2.imwrite(os.path.join(best_epe_image1_dir, "img1_{:06d}.png".format(id)), img1)
    cv2.imwrite(os.path.join(best_epe_image2_dir, "img2_{:06d}.png".format(id)), img2)
    cv2.imwrite(os.path.join(best_epe_flow_gt_dir, "gt_{:06d}.png".format(id)), flow_gt)
    cv2.imwrite(os.path.join(best_epe_flow_dcv_dir, "dcv_{:06d}.png".format(id)), flow_dcv)
    cv2.imwrite(os.path.join(best_epe_flow_ff_dir, "ff_{:06d}.png".format(id)), flow_ff)