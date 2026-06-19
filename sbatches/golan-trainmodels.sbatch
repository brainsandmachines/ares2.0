#!/bin/bash
#SBATCH --job-name=l2_16_init1
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/golan_neuro/convnextsmall/%j.out



nvidia-smi
source /home/ashtomer/projects/ares/sbatches/train_launcher_lib.sh
source /home/ashtomer/projects/ares/sbatches/stage_runner.sh
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

MODELS_ROOT="${MODELS_ROOT:-/home/ashtomer/projects/ares/results/models}"
MODEL_NAME_SUFFIX="${MODEL_NAME_SUFFIX:-}"
model_name="${model_name}${MODEL_NAME_SUFFIX}"

LRB=1.0e-3
if [[ "$attack_eps" == "8" ]]; then
    LRB=5.0e-4
elif [[ "$attack_eps" == "16" ]]; then
    LRB=2.5e-4
fi
LRB="${TRAIN_LRB:-$LRB}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-$BATCH_SIZE}"
TRAIN_RESUME="${TRAIN_RESUME:-auto}"
INIT_FROM_MODEL="${INIT_FROM_MODEL:-}"
INIT_FROM_CKPT="${INIT_FROM_CKPT:-model_best.pth.tar}"
GRADNORM_OBJECTIVE="${GRADNORM_OBJECTIVE:-}"
GRADNORM_MAX_REG_TO_CE_RATIO="${GRADNORM_MAX_REG_TO_CE_RATIO:-}"


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
echo "lrb            : $LRB"
echo "models_root    : $MODELS_ROOT"
echo "train_resume   : $TRAIN_RESUME"
echo "init_from_model: $INIT_FROM_MODEL"
echo "init_from_ckpt : $INIT_FROM_CKPT"
echo "gn_objective   : $GRADNORM_OBJECTIVE"
echo "gn_max_reg_ce  : $GRADNORM_MAX_REG_TO_CE_RATIO"

RESUME_PATH="${MODELS_ROOT}/${model_name}/last.pth.tar"
if [[ "$TRAIN_RESUME" == "none" ]]; then
    RESUME_PATH=""
elif [[ "$TRAIN_RESUME" != "auto" ]]; then
    echo "[ERROR] TRAIN_RESUME must be 'auto' or 'none', got: $TRAIN_RESUME" >&2
    exit 1
fi

cmd=(
    python -m robust_training.adversarial_training
    attacks.attack_norm="$attack_norm"
    attacks.advtrain="$advtrain"
    attacks.gradnorm="$gradnorm"
    attacks.gradnorm_penalty_norm="$gradnorm_penalty_norm"
    attacks.attack_criterion="$crit"
    attacks.attack_domain="$attack_domain"
    model.experiment_num="$experiment_num"
    model.experiment_name="$model_name"
    model.resume="$RESUME_PATH"
    training.batch_size=$BATCH_SIZE
    lr_scheduler.lrb="$LRB"
    hydra.run.dir="${MODELS_ROOT}/$model_name/${now:%Y-%m-%d}/${now:%H-%M-%S}"
)

if [[ -n "$TRAIN_EPOCHS" ]]; then
    cmd+=("training.epochs=$TRAIN_EPOCHS")
fi

if [[ -n "$INIT_FROM_MODEL" ]]; then
    cmd+=(
        "continuation.enabled=true"
        "continuation.checkpoint_path=${MODELS_ROOT}/${INIT_FROM_MODEL}/${INIT_FROM_CKPT}"
        "continuation.use_ema=true"
    )
fi

if [[ -n "$GRADNORM_OBJECTIVE" ]]; then
    cmd+=("attacks.gradnorm_objective=$GRADNORM_OBJECTIVE")
fi

if [[ -n "$GRADNORM_MAX_REG_TO_CE_RATIO" ]]; then
    cmd+=("attacks.gradnorm_max_reg_to_ce_ratio=$GRADNORM_MAX_REG_TO_CE_RATIO")
fi

if [[ "$advtrain" == "true" ]] && {
      [[ "$crit" == "madry" && "$attack_domain" == "pixel" ]] ||
      [[ "$crit" == "trades" && "$attack_domain" == "v1_feature" && "$GPU_TOTAL" -gt 90000 ]]
  }; then
      cmd+=("model.compile_model=True")
      echo "compile_model  : True"
  fi


if [[ -n "$model_override" ]]; then
    cmd+=("$model_override")
    cmd+=("model.v1_noise_mode=$v1_noise_mode")
fi

if [[ "$is_v1" == "true" && "$advtrain" == "true" ]]; then
    cmd+=("attacks.v1_attack_eps=$attack_eps")
elif [[ -n "$attack_eps" ]]; then
    cmd+=("attacks.attack_eps=$attack_eps")
fi

# Walk the orchestrator stage machine (TRAIN -> AA_SWEEP -> PLOT_RESULTS).
# Falls back to plain training when not orchestrated (ORCH_MODEL_ID unset).
orch_pipeline "${cmd[@]}"

status=$?
echo "[INFO] Pipeline exit status: $status"

if [ "$status" -eq 0 ]; then
    echo "[INFO] Running checkpoint pruning for $model_name"
    python /home/ashtomer/projects/ares/results/models/prune_model_checkpoints.py \
        "${MODELS_ROOT}/$model_name" \
        --keep-n 10
    echo "[INFO] Checkpoint pruning finished with status $?"
else
    echo "[INFO] Skipping checkpoint pruning because training failed or was killed"
fi

exit "$status"
