#!/bin/bash


# BASE_SCENE_DIR=/netscratch/vemburaj/Novel_View_Synthesis/datasets/TartanAir-sample/office/Hard
# BASE_SAVE_DIR=/netscratch/vemburaj/DCV-Net/Visualizations/TartanAir-sample/office/Hard

# BASE_SCENE_DIR=/netscratch/vemburaj/datasets/spring_sample/train
# BASE_SAVE_DIR=/netscratch/vemburaj/DCV-Net/Visualizations/spring_sample/train

BASE_SCENE_DIR=/home/vyadav/mnt/netscratch/datasets/kubrik-nk-optical-flow/rgba_1k/rgba/1k
BASE_SAVE_DIR=/home/vyadav/mnt/netscratch/DCV-Net/Visualizations/kubrik-nk-optical-flow/1k

# BASE_SCENE_DIR=/netscratch/vemburaj/datasets/mpi_sintel/raw/training/final
# BASE_SAVE_DIR=/netscratch/vemburaj/DCV-Net/Visualizations/mpi_sintel/final

# BASE_SCENE_DIR=/netscratch/vemburaj/datasets/rgbd_bonn_dataset
# BASE_SAVE_DIR=/netscratch/vemburaj/DCV-Net/Visualizations/rgbd_bonn_dataset

# BASE_SCENE_DIR=/netscratch/vemburaj/datasets/h2o3d_v1/evaluation
# BASE_SAVE_DIR=/netscratch/vemburaj/DCV-Net/Visualizations/h2o3d_v1/evaluation_stride_5


# Define your list of scene subdirectories
# SCENES=("rgbd_bonn_static")

# Automatically collect subdirectories
SCENES=($(find "$BASE_SCENE_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))
# SCENES=($(find "$BASE_SCENE_DIR" -mindepth 1 -maxdepth 1 -type d | grep -E '/rgbd_bonn_[p-r].*' | xargs -n1 basename | sort))


#for i in $(seq -w 00 29); do
#    SEQ="0$i"
#    echo "Processing $SEQ..."

for SEQ in "${SCENES[@]}"; do
    echo "Processing $SEQ..."

    python -u demo_video.py \
        --scene_dir ${BASE_SCENE_DIR}/${SEQ} \
        --save_dir ${BASE_SAVE_DIR}/${SEQ} \
        --save_flow_gt \
        --save_flow_pred \
        --save_flow_pred_with_epe \
        --stage things-ablation-dynamic-wo-tile \
        --name raft-things-lookup-softmax-noise-twins-att-nh-1-nl-1-nu-6-ns-1-nr-1-sc-first-gelu-full-postln-qk-dm-res-432-960-lr-0.00025-ns-120k \
        --ckpt_step_idx -1 \
        --dataset kubrik-nk \
        --encoder twins \
        --corr_radius 4 --lookup_softmax \
        --embedding_dim 256 \
        --att_nhead 1 \
        --att_layer_layout self cross \
        --att_layer_type full \
        --att_weight_share_after 6 \
        --att_fix_n_updates \
        --att_update_stride 1 \
        --att_n_repeats 1 \
        --att_layer_norm post \
        --att_use_mlp \
        --iter_sintel 32 \
        --iter_kitti 24 \
        --sintel_tile_sigma 0.05 \
        --att_activation GELU \
        --dynamic_motion_encoder \
        --att_share_qk_proj \
        --att_first_no_share \
        --swin_att_num_splits 2 \
        --dynamic_matching \
        --att_raft
done
