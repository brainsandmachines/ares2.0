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
    attacks.attack_norm=linf \
    attacks.attack_eps=5 \
    attacks.advtrain=True \
    attacks.gradnorm=True \
    model.experiment_num=1 \
    model.resume="/home/ashtomer/projects/ares/results/last.pth.tar" \
    model.experiment_num=1 \
    training.epochs=250 \
    training.batch_size=256 \
    optimizer.clip_grad=null \
    dataset.mixup_prob=1e-6 \
    dataset.train_dir="/storage/test/bml_group/tomerash/datasets/imagenet_sample/train/" \
    dataset.eval_dir="/storage/test/bml_group/tomerash/datasets/imagenet_sample/val/" \
    hydra.run.dir="/home/ashtomer/projects/ares/results/${now:%Y-%m-%d}/${now:%H-%M-%S}" 