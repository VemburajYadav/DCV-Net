#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name eval
#SBATCH --nodes 1
#SBATCH --partition A100-40GB
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=20G
#SBATCH --time=03-00:00:00

srun -K -N 1 \
     --container-mounts=/netscratch/vemburaj:/netscratch/vemburaj,/ds-av:/ds-av:ro,"`pwd`":"`pwd`" \
     --container-image=/netscratch/vemburaj/ENROOT_IMAGES/nvcr.io_nvidia_pytorch_22.11-py3-torchdata-einops-cudacorr.sqsh \
     --container-workdir="`pwd`" \
python -u evaluate.py --name raft-things-lookup-softmax-noise-twins-lr-0.00025-ns-120k \
                      --dataset kitti_tile sintel_tile \
                      --ckpt_step_idx -1 \
                      --encoder twins \
                      --corr_radius 4 --lookup_softmax \
                      --embedding_dim 256 \
                      --att_nhead 1 \
                      --att_layer_layout self cross \
                      --att_layer_type full \
                      --att_weight_share_after 3 \
                      --att_fix_n_updates \
                      --att_update_stride 1 \
                      --att_n_repeats 2 \
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

