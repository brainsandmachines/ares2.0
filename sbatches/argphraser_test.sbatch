#!/bin/bash
#SBATCH --job-name=convnext_small_training_hydra
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/golan_neuro/convnextsmall/%A/%a.out

nvidia-smi
# ---- Env once ----
module load anaconda
source activate tomer_advtrain
# ------------------

cd /home/ashtomer/projects/ares

export MASTER_PORT=$((10000 + RANDOM % 50000))

torchrun --nproc_per_node=1 -m robust_training.adversarial_training --config=/home/ashtomer/projects/ares/robust_training/train_configs/convnext_small.yaml \
    --advtrain=false \
    --epochs=150 \