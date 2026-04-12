#!/bin/bash
#SBATCH --job-name=madry_train_acc
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=0-27%8
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/madry_train_acc/%A/%a.out

set -euo pipefail

module load anaconda
source activate tomer_train

REPO_ROOT="${REPO_ROOT:-/home/ashtomer/projects/ares}"
TRAIN_DIR="${TRAIN_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/train}"
MODELS_ROOT="${MODELS_ROOT:-/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data_analysis/train_accuracy_eval}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/outs/madry_train_acc}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_MODELS="${MAX_MODELS:-}"
MAX_BATCHES="${MAX_BATCHES:-}"
USE_EMA="${USE_EMA:-1}"

TARGET_MODELS=(
  convnext_small_linf_8_init1
  convnext_small_linf_8_init2
  convnext_small_linf_16_init1
  convnext_small_linf_16_init2
  convnext_small_linf_2_init1
  convnext_small_linf_2_init2
  convnext_small_linf_4_init1
  convnext_small_linf_4_init2
  convnext_small_l2_2_init1
  convnext_small_l2_2_init2
  convnext_small_l2_4_init1
  convnext_small_l2_4_init2
  convnext_small_l2_8_init1
  convnext_small_l2_8_init2
  convnext_small_l2_8_init3
  convnext_small_l2_16_init1
  convnext_small_l2_16_init2
  convnext_small_l2_16_init3
  convnext_small_l1_2_init1
  convnext_small_l1_2_init2
  convnext_small_l1_4_init1
  convnext_small_l1_4_init2
  convnext_small_l1_8_init1
  convnext_small_l1_8_init2
  convnext_small_l1_8_init3
  convnext_small_l1_16_init1
  convnext_small_l1_16_init2
  convnext_small_l1_16_init3
)

NUM_MODELS=${#TARGET_MODELS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [[ ${TASK_ID} -ge ${NUM_MODELS} ]]; then
  echo "[INFO] SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range (num_models=${NUM_MODELS}). Exiting."
  exit 0
fi

MODEL_NAME="${TARGET_MODELS[$TASK_ID]}"

GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
if [[ -z "${EVAL_BATCH_SIZE:-}" ]]; then
  if [[ ${GPU_TOTAL} -ge 95000 ]]; then
    EVAL_BATCH_SIZE=512
  elif [[ ${GPU_TOTAL} -ge 47000 ]]; then
    EVAL_BATCH_SIZE=256
  elif [[ ${GPU_TOTAL} -ge 23000 ]]; then
    EVAL_BATCH_SIZE=128
  elif [[ ${GPU_TOTAL} -ge 11000 ]]; then
    EVAL_BATCH_SIZE=32
  elif [[ ${GPU_TOTAL} -ge 5000 ]]; then
    EVAL_BATCH_SIZE=32
  else
    EVAL_BATCH_SIZE=16
  fi
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

cd "${REPO_ROOT}"

echo "[INFO] num_target_models=${NUM_MODELS}"
echo "[INFO] selected_model=${MODEL_NAME}"
echo "[INFO] detected_gpu_total_mb=${GPU_TOTAL}"
echo "[INFO] eval_batch_size=${EVAL_BATCH_SIZE}"

scontrol update JobId=${SLURM_ARRAY_JOB_ID}_${TASK_ID} \
  JobName=madry_acc_${MODEL_NAME} || true

CMD=(
  python data_analysis/eval_madry_train_accuracy.py
  --models-dir "${MODELS_ROOT}"
  --train-dir "${TRAIN_DIR}"
  --out-csv "${OUT_ROOT}/${MODEL_NAME}.csv"
  --log-path "${LOG_ROOT}/${MODEL_NAME}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}-${TASK_ID}.log"
  --device cuda
  --batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --model-name "${MODEL_NAME}"
)

if [[ -n "${MAX_MODELS}" ]]; then
  CMD+=(--max-models "${MAX_MODELS}")
fi

if [[ -n "${MAX_BATCHES}" ]]; then
  CMD+=(--max-batches "${MAX_BATCHES}")
fi

if [[ "${USE_EMA}" == "0" ]]; then
  CMD+=(--no-use-ema)
fi

printf '[INFO] %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
