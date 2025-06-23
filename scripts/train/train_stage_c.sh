#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name bash
#SBATCH --nodes 1
#SBATCH --partition A100-40GB
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
python -u train.py --name raft-chairs-noise-b-10-lr-0.0004-ns-120k \
                   --encoder cnn \
                   --add_noise \
                   --stage chairs --validation chairs --gpus 0 1 \
                   --num_steps 120000 --batch_size 10 --lr 0.0004 --warmup_steps 6000 \
                   --image_size 368 496 --wdecay 0.0001 --corr_radius 4 \
                   --iters 12 \
                   --scheduler OneCycleLR \
                   --att_nhead 1 \
                   --att_layer_layout self cross \
                   --att_layer_type full \
                   --att_layer_norm post \
                   --att_weight_share_after 3 \
                   --att_fix_n_updates \
                   --att_update_stride 1 \
                   --att_n_repeats 2 \
                   --att_use_mlp \
                   --att_activation GELU \
                   --embedding_dim 256 \
                   --att_first_no_share \
                   --att_share_qk_proj \
                   --swin_att_num_splits 2 \
                   --dynamic_motion_encoder
#                   --dynamic_matching \

















































#                   --coarse_supervision \
#                   --coarse_loss_weight 0.001

#                   --dynamic_coarse_supervision \
#                   --coarse_loss_weight 0.004
#                   --dynamic_motion_encoder
#                   --dynamic_coarse_supervision \
#                   --coarse_loss_weight 0.001

    # raft-chairs-lookup-softmax-all-nh-1-nl-1-linear-share_x6-fix-lr-0.0004-ns-100k
