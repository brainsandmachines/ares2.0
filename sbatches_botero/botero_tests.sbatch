#!/bin/bash
#SBATCH --job-name=convnext_small_training_adv
#SBATCH --time=7-00:00:00
#SBATCH --partition=main                 # main queue
#SBATCH --gpus=1                          # request 1 GPU         
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/main/convnextsmall/%A/%a.out
#SBATCH --array=1-200%1


nvidia-smi
# ---- Env once ----
module load anaconda
source activate tomer_advtrain_pro
# ------------------

cd /home/ashtomer/projects/ares

python -c "import torch; print(torch.version.cuda, torch.version.git_version)"


export MASTER_PORT=$((10000 + RANDOM % 50000))

model_name="convnext_small_l1_2_init1"

python -m robust_training.adversarial_training \
    attacks.attack_eps=3 \
    attacks.advtrain=false \
    attacks.attack_norm=linf \
    attacks.gradnorm=true \
    model.experiment_num=12 \
    training.batch_size=40 \
    dataset.train_dir=/mnt/data/datasets/imagenet_sample/train \
    dataset.eval_dir=/mnt/data/datasets/imagenet_sample/val \