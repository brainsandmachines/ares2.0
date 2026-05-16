#!/bin/bash
#SBATCH --job-name=l2_16_init1
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=1-10%1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/golan_neuro/convnextsmall/%A/%a.out



nvidia-smi
source /home/ashtomer/projects/ares/sbatches/train_launcher_lib.sh
# -----------------------------
# 1. Get GPU Memory (in MB)
# -----------------------------
# This extracts the number (e.g., "48000") from nvidia-smi
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[INFO] Detected GPU Memory: ${GPU_TOTAL} MB"
# ---- Env once ----
module load anaconda
source activate "$(select_train_env "$GPU_TOTAL")"
# ------------------


cd /home/ashtomer/projects/ares

python -c "import torch; print(torch.version.cuda, torch.version.git_version)"

export MASTER_PORT=$((10000 + RANDOM % 50000))

jobname="$SLURM_JOB_NAME"

parse_train_job "$jobname" "$GPU_TOTAL" || exit 1


echo "JOB NAME        : $jobname"
echo "job_core        : $job_core"
echo "model_size      : $model_size"
echo "attack_eps     : $attack_eps"
echo "experiment_num : $experiment_num"
echo "attack_norm    : $attack_norm"
echo "advtrain       : $advtrain"
echo "gradnorm       : $gradnorm"
echo "gradnorm_pen   : $gradnorm_penalty_norm"
echo "is_v1          : $is_v1"
echo "v1_noise_mode  : $v1_noise_mode"
echo "attack_domain  : $attack_domain"
echo "model_name     : $model_name"
echo "Batch size     : $BATCH_SIZE"
echo "attack_crit    : $crit"

cmd=(
    python -m robust_training.adversarial_training
    attacks.attack_norm="$attack_norm"
    attacks.advtrain="$advtrain"
    attacks.gradnorm="$gradnorm"
    attacks.gradnorm_penalty_norm="$gradnorm_penalty_norm"
    attacks.attack_criterion="$crit"
    attacks.attack_domain="$attack_domain"
    model.experiment_num="$experiment_num"
    model.resume="/home/ashtomer/projects/ares/results/models/$model_name/last.pth.tar"
    training.batch_size=$BATCH_SIZE
    hydra.run.dir="/home/ashtomer/projects/ares/results/models/$model_name/${now:%Y-%m-%d}/${now:%H-%M-%S}"
)

if [[ -n "$model_override" ]]; then
    cmd+=("$model_override")
    cmd+=("model.v1_noise_mode=$v1_noise_mode")
fi

if [[ "$is_v1" == "true" && "$advtrain" == "true" ]]; then
    cmd+=("attacks.v1_attack_eps=$attack_eps")
elif [[ -n "$attack_eps" ]]; then
    cmd+=("attacks.attack_eps=$attack_eps")
fi

"${cmd[@]}"

status=$?
echo "[INFO] Training exit status: $status"

if [ "$status" -eq 0 ]; then
    echo "[INFO] Running checkpoint pruning for $model_name"
    python /home/ashtomer/projects/ares/results/models/prune_model_checkpoints.py \
        "/home/ashtomer/projects/ares/results/models/$model_name" \
        --keep-n 10
    echo "[INFO] Checkpoint pruning finished with status $?"
else
    echo "[INFO] Skipping checkpoint pruning because training failed or was killed"
fi

exit "$status"
