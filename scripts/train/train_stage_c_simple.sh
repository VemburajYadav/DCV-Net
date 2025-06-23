#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name bash
#SBATCH --nodes 1
#SBATCH --partition A100-40GB
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gpus-per-task 2
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03-00:00:00

srun -K -N 1 \
     --container-mounts=/netscratch/vemburaj:/netscratch/vemburaj,/ds-av:/ds-av:ro,"`pwd`":"`pwd`" \
     --container-image=/netscratch/vemburaj/ENROOT_IMAGES/nvcr.io_nvidia_pytorch_22.11-py3-torchdata-einops-cudacorr.sqsh \
     --container-workdir="`pwd`" \
python -u train.py --name raft-chairs-lookup-softmax-coarse-ce-1.0-lr-0.0004-ns-120k \
                   --encoder cnn \
                   --stage chairs --validation chairs --gpus 0 1 \
                   --num_steps 120000 --batch_size 10 --lr 0.0004 --warmup_steps 5000 \
                   --image_size 368 496 --wdecay 0.0001 --corr_radius 4 --lookup_softmax \
                   --scheduler OneCycleLR \
                   --coarse_supervision \
                   --coarse_loss_weight 1.0



