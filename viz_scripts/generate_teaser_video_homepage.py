import os
import numpy as np
import cv2
import sys
from pathlib import Path
from glob import glob
from core.utils.frame_utils import read_gen
from core.utils.cv2_viz_utils import viz_img_grid


data_root = "/home/vyadav/mnt/netscratch/DCV-Net/Visualizations"
# data_root = "/netscratch/vemburaj/DCV-Net/Visualizations"
data_root = Path(data_root)

SCENE = "h2o3d_v1/evaluation_stride_5/SSPD3"
SCENE_DIR = data_root / SCENE

SAVE_DIR = data_root / "Teaser_Videos" / "/".join(SCENE.split("/")[:-1])
SAVE_PATH = SAVE_DIR / (SCENE.split("/")[-1] + ".mp4")

print("SCENE_DIR: ", SCENE_DIR)
print("SAVE_PATH: ", SAVE_PATH)

if __name__ == '__main__':
    img1_dir = SCENE_DIR / "image1"
    flow_dir = SCENE_DIR / "things" / "flow_pred"

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    image_paths = list(sorted(glob(os.path.join(str(img1_dir), "*.jpg"))))
    flow_paths = list(sorted(glob(os.path.join(str(flow_dir), "*.jpg"))))

    img1 = np.array(read_gen(image_paths[0])).astype(np.uint8)
    flow = np.array(read_gen(flow_paths[0])).astype(np.uint8)
    viz_frame = viz_img_grid(data_grid=[[img1, flow]], spacing=10)

    # Example for 10 frames
    video = cv2.VideoWriter(str(SAVE_PATH), cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (viz_frame.shape[1], viz_frame.shape[0]))

    for i in range(len(image_paths)):
        print(f"Processing frame {i}, Total frames: {len(image_paths)}")
        img1 = np.array(read_gen(image_paths[i])).astype(np.uint8)
        flow = np.array(read_gen(flow_paths[i])).astype(np.uint8)

        viz_frame = viz_img_grid(data_grid=[[img1, flow]], spacing=10)
        video.write(viz_frame)

    video.release()








