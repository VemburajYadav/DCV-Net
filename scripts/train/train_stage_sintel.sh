#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name bash
#SBATCH --nodes 1
#SBATCH --partition A100-80GB
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gpus-per-task 2
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03-00:00:00

srun -K -N 1 \
     --container-mounts=/netscratch/vemburaj:/netscratch/vemburaj,/ds-av:/ds-av:ro,"`pwd`":"`pwd`" \
     --container-image=/netscratch/vemburaj/ENROOT_IMAGES/nvcr.io_nvidia_pytorch_22.11-py3-torchdata-einops-cudacorr.sqsh \
     --container-workdir="`pwd`" \
python -u train.py --name raft-kitti-lookup-softmax-noise-twins-nh-1-nl-1-nu-3-ns-1-nr-2-sc-first-gelu-full-postln-qk-dm-res-432-960-lr-0.00025-ns-160k \
                   --encoder twins --pretrain_imagenet \
                   --add_noise \
                   --validation none \
                   --stage kitti \
                   --restore_ckpt raft-sintel-lookup-softmax-noise-twins-nh-1-nl-1-nu-3-ns-1-nr-2-sc-first-gelu-full-postln-qk-dm-res-432-960-lr-0.00025-ns-200k \
                   --restore_ckpt_step_idx -1 \
                   --gpus 0 1 --num_steps 160000 --warmup_steps 8000 \
                   --batch_size 6 --lr 0.00025  \
                   --image_size 432 960 --wdecay 0.00001 --gamma 0.85 \
                   --corr_radius 4 --lookup_softmax \
                   --embedding_dim 256 \
                   --scheduler OneCycleLR \
                   --dynamic_matching \
                   --att_nhead 1 \
                   --att_layer_layout self cross \
                   --att_layer_type full \
                   --att_weight_share_after 3 \
                   --att_fix_n_updates \
                   --att_update_stride 1 \
                   --att_n_repeats 2 \
                   --att_layer_norm post \
                   --att_activation GELU \
                   --att_use_mlp \
                   --dynamic_motion_encoder \
                   --att_share_qk_proj \
                   --att_first_no_share \
                   --swin_att_num_splits 2
#                   --coarse_supervision \
#                   --coarse_loss_weight 0.001




#                   --dynamic_coarse_supervision \
#                   --coarse_loss_weight 0.05


































































































































































































































































