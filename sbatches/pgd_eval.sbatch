#!/bin/bash
#SBATCH --job-name=PGD_eval_trades
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --array=0-15%3
#SBATCH --output=/home/ashtomer/projects/ares/outs/PGD_eval/%A/%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_train

cd /home/ashtomer/projects/ares

# =========================
# Models with >=199 epochs and missing PGD eval
# =========================
MODELS_ROOT="/home/ashtomer/projects/ares/results/models"
TARGET_MODELS=(
    "convnext_small_baseline_init1"
    "convnext_small_baseline_init2"
    "convnext_small_l2trades_2_init1"
    "convnext_small_l2trades_2_init2"
    "convnext_small_l2trades_4_init1"
    "convnext_small_l2trades_4_init2"
    "convnext_small_l2trades_8_init1"
    "convnext_small_l2trades_8_init2"
    "convnext_small_linftrades_1_init1"
    "convnext_small_linftrades_1_init2"
    "convnext_small_linftrades_2_init1"
    "convnext_small_linftrades_2_init2"
    "convnext_small_linftrades_4_init1"
    "convnext_small_linftrades_4_init2"
    "convnext_small_linftrades_8_init1"
    "convnext_small_linftrades_8_init2"
)
NUM_MODELS=${#TARGET_MODELS[@]}

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
VAL_DIR="/groups/golan_neurogroup/bml_group/tomerash/datasets/imagenet/val/"
OUT_DIR="${MODEL_DIR}/pgd_eval"

echo "[INFO] num_target_models=${NUM_MODELS}"
echo "[INFO] selected_model=${MODEL_NAME}"
echo "[INFO] model_dir=${MODEL_DIR}"
echo "[INFO] ckpt=${CKPT}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}"
  exit 1
fi

# Rename current array task for visibility in squeue (non-fatal if update races)
scontrol update JobId=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
  JobName=PGD_eval_${MODEL_NAME} || true

mkdir -p "${OUT_DIR}"

# =========================
# Run eval
# =========================
python data_analysis/final_eval.py \
  --checkpoint "${CKPT}" \
  --val-dir "${VAL_DIR}" \
  --pgd --plots \
  --device cuda \
  --out-dir "${OUT_DIR}" \
  --pgd-batch-size 256
