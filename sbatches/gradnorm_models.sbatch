#!/bin/bash
#SBATCH --job-name=gradnorm_1_init2
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

model_name="convnext_small_gradnorm_1_init2"

python -m robust_training.adversarial_training \
    attacks.attack_eps=1 \
    attacks.advtrain=false \
    attacks.attack_norm=linf \
    attacks.gradnorm=true \
    model.experiment_num=1 \
    model.resume="/home/ashtomer/projects/ares/results/models/convnext_small_gradnorm_1_init2/last.pth.tar" \
    training.epochs=300 \
    training.batch_size=256 \
    optimizer.clip_grad=null \
    dataset.mixup_active=True \
    hydra.run.dir="/home/ashtomer/projects/ares/results/models/convnext_small_gradnorm_1_init2/${now:%Y-%m-%d}/${now:%H-%M-%S}" 