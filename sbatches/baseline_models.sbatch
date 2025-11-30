#!/bin/bash
#SBATCH --job-name=baseline_init1
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

model_name="convnext_small_baseline_init1"

python -m robust_training.adversarial_training \
    attacks.attack_eps=0 \
    attacks.advtrain=false \
    attacks.attack_norm=linf \
    attacks.gradnorm=false \
    model.experiment_num=1 \
    model.resume="/home/ashtomer/projects/ares/results/models/$model_name/last.pth.tar" \
    training.epochs=300 \
    training.batch_size=256 \
    hydra.run.dir="/home/ashtomer/projects/ares/results/models/$model_name/${now:%Y-%m-%d}/${now:%H-%M-%S}" 