#!/bin/bash
#SBATCH --job-name=PGD_eval_trades
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --dependency=afterany:15306484
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --array=0-22%8
#SBATCH --output=/home/ashtomer/projects/ares/outs/PGD_eval/%A/%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_train

cd /home/ashtomer/projects/ares

# =========================
# Fixed sweep: Madry models (from check_models_status.py) that reached >=199
# =========================
MODELS_ROOT="/home/ashtomer/projects/ares/results/models"
TARGET_MODELS=(
  "convnext_small_linf_1_init2"
  "convnext_small_linf_2_init1"
  "convnext_small_linf_2_init2"
  "convnext_small_linf_4_init1"
  "convnext_small_linf_4_init2"
  "convnext_small_linf_8_init1"
  "convnext_small_linf_8_init2"
  "convnext_small_linf_16_init1"
  "convnext_small_linf_16_init2"
  "convnext_small_l2_1_init1"
  "convnext_small_l2_1_init2"
  "convnext_small_l2_2_init1"
  "convnext_small_l2_2_init2"
  "convnext_small_l2_4_init1"
  "convnext_small_l2_4_init2"
  "convnext_small_l2_8_init1"
  "convnext_small_l2_8_init2"
  "convnext_small_l2_16_init1"
  "convnext_small_l2_16_init2"
  "convnext_small_l1_1_init1"
  "convnext_small_l1_1_init2"
  "convnext_small_l1_2_init1"
  "convnext_small_l1_2_init2"
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
VAL_DIR="/storage/test/bml_group/tomerash/datasets/imagenet/val/"
OUT_DIR="${MODEL_DIR}/pgd_eval"

echo "[INFO] num_target_models=${NUM_MODELS}"
echo "[INFO] selected_model=${MODEL_NAME}"
echo "[INFO] model_dir=${MODEL_DIR}"
echo "[INFO] ckpt=${CKPT}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}"
  exit 1
fi

# Rename job for visibility in squeue
scontrol update JobId=$SLURM_JOB_ID \
  JobName=PGD_eval_${MODEL_NAME}

mkdir -p "${OUT_DIR}"

# =========================
# Run eval
# =========================
python data_analysis/final_eval.py \
  --checkpoint "${CKPT}" \
  --val-dir "${VAL_DIR}" \
  --pgd --plots \
  --device cuda \
  --out-dir "${OUT_DIR}"
