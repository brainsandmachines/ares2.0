#!/bin/bash
#SBATCH --job-name=linf_2_init1
#SBATCH --time=7-00:00:00
#SBATCH --partition=main                 # main queue
#SBATCH --gpus=1                          # request 1 GPU
#SBATCH --constraint=rtx_pro_6000            # GPUs with >24GB VRAM
#SBATCH --mem=96G                        # memory (optional override)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/main/convnextsmall/%A/%a.out
#SBATCH --array=1-100%1


nvidia-smi
# ---- Env once ----
module load anaconda
source activate tomer_advtrain_pro
# ------------------

cd /home/ashtomer/projects/ares

python -c "import torch; print(torch.version.cuda, torch.version.git_version)"


export MASTER_PORT=$((10000 + RANDOM % 50000))

model_name="convnext_small_linf_2_init1"

python -m robust_training.adversarial_training \
    attacks.attack_eps=2 \
    attacks.advtrain=true \
    attacks.attack_norm=linf \
    attacks.gradnorm=false \
    model.experiment_num=1 \
    model.resume="/home/ashtomer/projects/ares/results/models/${model_name}/last.pth.tar" \
    training.epochs=300 \
    training.batch_size=512 \
    hydra.run.dir="/home/ashtomer/projects/ares/results/models/${model_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}" 