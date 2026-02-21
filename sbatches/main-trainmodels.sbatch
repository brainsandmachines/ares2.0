#!/bin/bash
#SBATCH --job-name=l2trades_1_init1
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

jobname="$SLURM_JOB_NAME"

# -----------------------------
# Parse attack_eps (number before init)
# -----------------------------
attack_eps=$(echo "$jobname" | sed -E 's/.*_([0-9]+)_init[0-9]+/\1/')

# -----------------------------
# Parse experiment_num (number after init)
# -----------------------------
experiment_num=$(echo "$jobname" | sed -E 's/.*_init([0-9]+)/\1/')

# -----------------------------
# Defaults
# -----------------------------
attack_norm="linf"
advtrain=false
gradnorm=false
crit="madry"

# -----------------------------
# Rules based on job name
# -----------------------------
if [[ "$jobname" == *gradnorm* ]]; then
    gradnorm=true
elif [[ "$jobname" == *linf* ]]; then
    attack_norm="linf"
    advtrain=true
elif [[ "$jobname" == *l2* ]]; then
    attack_norm="l2"
    advtrain=true
fi

if [[ "$jobname" == *trades* ]]; then
    crit="trades"
fi

# -----------------------------
# 1. Get GPU Memory (in MB)
# -----------------------------
# This extracts the number (e.g., "48000") from nvidia-smi
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[INFO] Detected GPU Memory: ${GPU_TOTAL} MB"

# -----------------------------
# 2. Set Batch Size
# -----------------------------
if [ "$GPU_TOTAL" -ge 90000 ]; then
    # A100 (80GB) range
    if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
        BATCH_PER_GPU=384
    else
        BATCH_PER_GPU=512
    fi
elif [ "$GPU_TOTAL" -ge 45000 ]; then
    # RTX 6000 / A6000 (48GB) range
    if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
        BATCH_PER_GPU=192
    else
        BATCH_PER_GPU=256
    fi
elif [ "$GPU_TOTAL" -ge 22000 ]; then
    # 3090 / 4090 / A10 (24GB) range
    if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
        BATCH_PER_GPU=96
    else
        BATCH_PER_GPU=128
    fi
else
    echo "[ERROR] GPU total memory too small (< 22 GB). Detected: $GPU_TOTAL"
    exit 1
fi

BATCH_SIZE=$BATCH_PER_GPU

# -----------------------------
# Model name
# -----------------------------
model_name="convnext_small_${jobname}"

echo "JOB NAME        : $jobname"
echo "attack_eps     : $attack_eps"
echo "experiment_num : $experiment_num"
echo "attack_norm    : $attack_norm"
echo "advtrain       : $advtrain"
echo "gradnorm       : $gradnorm"
echo "model_name     : $model_name"
echo "Batch size     : $BATCH_SIZE"
echo "attack_crit    : $crit"

python -m robust_training.adversarial_training \
    attacks.attack_eps="$attack_eps" \
    attacks.attack_norm="$attack_norm" \
    attacks.advtrain="$advtrain" \
    attacks.gradnorm="$gradnorm" \
    attacks.attack_criterion="$crit" \
    model.experiment_num="$experiment_num" \
    model.resume="/home/ashtomer/projects/ares/results/models/$model_name/last.pth.tar" \
    training.batch_size=$BATCH_SIZE \
    hydra.run.dir="/home/ashtomer/projects/ares/results/models/$model_name/${now:%Y-%m-%d}/${now:%H-%M-%S}"