#!/bin/bash
#SBATCH --job-name=convnext_small_training
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/golan_neuro/convnextsmall/%A/%a.out
#SBATCH --array=1-500%1



nvidia-smi
# ---- Env once ----
module load anaconda
source activate tomer_advtrain
# ------------------

cd /home/ashtomer/projects/ares

python -c "import torch; print(torch.version.cuda, torch.version.git_version)"


export MASTER_PORT=$((10000 + RANDOM % 50000))

python -m robust_training.adversarial_training \
    attacks.attack_eps=1 \
    attacks.advtrain=false \
    attacks.attack_norm=linf \
    attacks.gradnorm=true \
    model.experiment_num=1 \
    model.resume="/home/ashtomer/projects/ares/results/models/convnext_small_gradnorm_1_init1/last.pth.tar" \
    training.epochs=300 \
    training.batch_size=128 \
    optimizer.clip_grad=null \
    dataset.mixup_active=True \
    hydra.run.dir="/home/ashtomer/projects/ares/results/models/convnext_small_gradnorm_1_init1/${now:%Y-%m-%d}/${now:%H-%M-%S}" 