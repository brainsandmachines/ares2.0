#!/bin/bash
#SBATCH --job-name=madry_train_acc
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/madry_train_acc/%A.out

set -euo pipefail

module load anaconda
source activate tomer_train

REPO_ROOT="${REPO_ROOT:-/home/ashtomer/projects/ares}"
TRAIN_DIR="${TRAIN_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/train}"
MODELS_ROOT="${MODELS_ROOT:-/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data_analysis/train_accuracy_eval}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/outs/madry_train_acc}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
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

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

cd "${REPO_ROOT}"

CMD=(
  python data_analysis/eval_madry_train_accuracy.py
  --models-dir "${MODELS_ROOT}"
  --train-dir "${TRAIN_DIR}"
  --out-csv "${OUT_ROOT}/madry_train_accuracy_selected.csv"
  --log-path "${LOG_ROOT}/madry_train_accuracy-${SLURM_JOB_ID:-manual}.log"
  --device cuda
  --batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)

for model_name in "${TARGET_MODELS[@]}"; do
  CMD+=(--model-name "${model_name}")
done

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
