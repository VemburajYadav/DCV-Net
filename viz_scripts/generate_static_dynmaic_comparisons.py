import os
import numpy as np
import cv2
import sys
from pathlib import Path
from glob import glob
from core.utils.frame_utils import read_gen
from core.utils.cv2_viz_utils import viz_img_grid, put_text_in_frame

WO_TILE = True
STAGE = "things"

TILED = ""
if WO_TILE:
    TILED = TILED + "-wo-tile"



data_root = "/home/vyadav/mnt/netscratch/DCV-Net/Visualizations"
# data_root = "/netscratch/vemburaj/DCV-Net/Visualizations"
data_root = Path(data_root)
SCENE_DIR = ("kubrik-nk-optical-flow/1k")
# SCENE_DIR = "mpi_sintel/final"

SAVE_DIR = data_root / "Static-Dynamic-Comparisons" / SCENE_DIR

SCENE_DIR = data_root / SCENE_DIR
SCENES = ["029"]

VIDEO_SAVE_PATH = SAVE_DIR / ("-".join(SCENES) + ".mp4")
SAVE_DIR = SAVE_DIR / "-".join(SCENES)

print("SCENE_DIR: ", SCENE_DIR)
# print("SAVE_PATH: ", SAVE_PATH)

if __name__ == '__main__':

    num_images = [len(list(sorted(glob(os.path.join(str(SCENE_DIR / scene / "image1"), "*.png"))))) for scene in SCENES]
    max_frames = np.array([num_images]).astype(np.int32).max()

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for i in range(max_frames):

        print(f"Processing frame {i}, Total frames: {max_frames}")
        data_grid = []

        for scene in SCENES:

            img1_dir = SCENE_DIR / scene / "image1"
            img2_dir = SCENE_DIR / scene / "image2"

            flow_dir_static = SCENE_DIR / scene / (STAGE + "-ablation-static" + TILED) / "flow_pred_with_epe"
            flow_dir_dynamic = SCENE_DIR / scene / (STAGE + "-ablation-dynamic" + TILED) / "flow_pred_with_epe"
            flow_dir_gt = SCENE_DIR / scene / "flow_gt"

            image1_paths = list(sorted(glob(os.path.join(str(img1_dir), "*.png"))))
            image2_paths = list(sorted(glob(os.path.join(str(img2_dir), "*.png"))))
            flow_paths_static = list(sorted(glob(os.path.join(str(flow_dir_static), "*.png"))))
            flow_paths_dynamic = list(sorted(glob(os.path.join(str(flow_dir_dynamic), "*.png"))))
            flow_paths_gt = list(sorted(glob(os.path.join(str(flow_dir_gt), "*.png"))))

            idx = i % len(image1_paths)
            print(f"{scene}: {idx}")
            img1 = np.array(read_gen(image1_paths[idx])).astype(np.uint8)
            img2 = np.array(read_gen(image2_paths[idx])).astype(np.uint8)
            flow_gt = np.array(read_gen(flow_paths_gt[idx])).astype(np.uint8)
            flow_static = np.array(read_gen(flow_paths_static[idx])).astype(np.uint8)
            flow_dynamic = np.array(read_gen(flow_paths_dynamic[idx])).astype(np.uint8)

            # data_grid += [[put_text_in_frame(img1, "Image 1", location="top-left", bg="dark"),
            #                put_text_in_frame(img2, "Image 2", location="top-left", bg="dark"),
            #                put_text_in_frame(flow_gt, "Ground Truth", location="top-left", bg="dark")]]
            # data_grid += [[put_text_in_frame(flow_dcv, "DCV-Net", location="top-left", bg="dark"),
            #                put_text_in_frame(flow_gmflow, "GMFlow", location="top-left", bg="dark"),
            #                put_text_in_frame(flow_ff, "FlowFormer", location="top-left", bg="dark")]]

            data_grid += [[put_text_in_frame(img1, "Image 1", location="top-left", bg="dark"),
                           put_text_in_frame(img2, "Image 2", location="top-left", bg="dark")]]
            data_grid += [[put_text_in_frame(flow_static, "Static CV", location="top-left", bg="dark"),
                           put_text_in_frame(flow_dynamic, "Dynamic CV", location="top-left", bg="dark")]]
            data_grid += [[put_text_in_frame(flow_gt, "Ground Truth", location="top-left", bg="dark")]]
            # data_grid.append([img1, img2, flow_gt, flow_dcv, flow_gmflow, flow_ff])

        viz_frame = viz_img_grid(data_grid=data_grid, spacing=10)
        cv2.imwrite(os.path.join(str(SAVE_DIR), f"{i:0>4}.png"), viz_frame)

        if i == 0:
            video = cv2.VideoWriter(str(VIDEO_SAVE_PATH), cv2.VideoWriter_fourcc(*'mp4v'), 1,
                                    (viz_frame.shape[1], viz_frame.shape[0]))

        video.write(viz_frame)
        video.write(viz_frame)

    video.release()








