#!/bin/bash
#SBATCH --job-name=PGD_eval_madry_l1_apgd
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --array=0-63%3
#SBATCH --output=/home/ashtomer/projects/ares/outs/PGD_eval_madry_l1_apgd/%A/%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_train

REPO_ROOT="${REPO_ROOT:-/home/ashtomer/projects/ares}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/results/models}"
VAL_DIR="${VAL_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/val/}"
OUT_SUBDIR="${OUT_SUBDIR:-pgd_eval_l1_apgd}"
OUT_CSV="${OUT_CSV:-pgd_validation_results_l1_apgd.csv}"
PGD_EPS="${PGD_EPS:-1,2,4,8,16}"
PGD_BATCH_SIZE="${PGD_BATCH_SIZE:-256}"
L1_APGD_RHO="${L1_APGD_RHO:-0.05}"

cd "${REPO_ROOT}"

mapfile -t TARGET_MODELS < <(
  find -L "${MODELS_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | grep -Ev '(^(__pycache__|old_models)$|trades|gradnorm)' \
    | sort
)

NUM_MODELS=${#TARGET_MODELS[@]}
if [[ ${NUM_MODELS} -eq 0 ]]; then
  echo "[ERROR] No eligible model directories found under ${MODELS_ROOT}"
  exit 1
fi

if [[ ${SLURM_ARRAY_TASK_ID} -ge ${NUM_MODELS} ]]; then
  echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range (num_models=${NUM_MODELS}). Exiting."
  exit 0
fi

MODEL_NAME="${TARGET_MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_DIR="${MODELS_ROOT}/${MODEL_NAME}"
CKPT="${MODEL_DIR}/model_best.pth.tar"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${MODEL_DIR}/model_best.pth"
fi
OUT_DIR="${MODEL_DIR}/${OUT_SUBDIR}"

echo "[INFO] num_target_models=${NUM_MODELS}"
echo "[INFO] selected_model=${MODEL_NAME}"
echo "[INFO] model_dir=${MODEL_DIR}"
echo "[INFO] ckpt=${CKPT}"
echo "[INFO] val_dir=${VAL_DIR}"
echo "[INFO] out_dir=${OUT_DIR}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[WARN] Checkpoint not found for ${MODEL_NAME}. Skipping."
  exit 0
fi

if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] Validation directory not found: ${VAL_DIR}"
  exit 1
fi

scontrol update JobId=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
  JobName=PGD_l1apgd_${MODEL_NAME} || true

mkdir -p "${OUT_DIR}"

python data_analysis/final_eval.py \
  --checkpoint "${CKPT}" \
  --val-dir "${VAL_DIR}" \
  --pgd \
  --plots \
  --device cuda \
  --out-dir "${OUT_DIR}" \
  --pgd-output-csv "${OUT_CSV}" \
  --pgd-norms l1 \
  --pgd-eps "${PGD_EPS}" \
  --pgd-batch-size "${PGD_BATCH_SIZE}" \
  --l1-step-mode l1_apgd \
  --l1-apgd-rho "${L1_APGD_RHO}"
