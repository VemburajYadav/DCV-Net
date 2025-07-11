#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name eval
#SBATCH -o ./logs/slurm_eval_logs/kitti-noise-twins-full-res-432-960-lr-0.00025-ns-50k-eval-kitti-sintel-all.out
#SBATCH --nodes 1
#SBATCH --partition A100-40GB,A100-PCI,RTX3090,H100,H100-RP,H100-PCI,A100-RP,RTXA6000,RTXA6000-AV,V100-16GB,V100-32GB,H200,A100-80GB
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
python -u evaluate.py --name raft-kitti-lookup-softmax-noise-twins-nh-1-nl-1-nu-3-ns-1-nr-2-sc-first-gelu-full-postln-qk-dm-res-432-960-lr-0.00025-ns-50k \
                      --ckpt_step_idx -1 \
                      --dataset kitti kitti_tile sintel sintel_tile \
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
                      --dynamic_matching

