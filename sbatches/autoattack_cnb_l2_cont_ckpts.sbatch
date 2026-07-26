#!/bin/bash
#SBATCH --job-name=aa_cnb_l2_linf6_ckpts
#SBATCH --time=1-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=0-38%6
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/autoattack_extra_checkpoints/cnb_l2_linf6/%A_%a.out

# AutoAttack, Linf norm, eps_input=6 (eval eps=6/255), 64 images in a single
# batch, over every checkpoint kept in each of the 3 convnext_base l2
# cont4to6 lr-grid model dirs: the 10 rolling checkpoint-N.pth.tar files plus
# last.pth.tar, model_best.pth.tar, and model_best_adv.pth.tar (13 per model,
# 39 total).

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

MODELS_ROOT=/home/ashtomer/projects/ares/results/models
VAL_DIR=/groups/golan_neurogroup/bml_group/datasets/imagenet/val
BATCH_SIZE=64

MODEL_DIRS=(
  "convnext_base_l2_cont4to6_lr1e4_init0"
  "convnext_base_l2_cont4to6_lr5e5_init0"
  "convnext_base_l2_cont4to6_lr25e6_init0"
)

CKPTS_LR1E4=(checkpoint-0 checkpoint-1 checkpoint-2 checkpoint-3 checkpoint-4 checkpoint-5 checkpoint-6 checkpoint-7 checkpoint-8 checkpoint-10 last model_best model_best_adv)
CKPTS_LR5E5=(checkpoint-0 checkpoint-1 checkpoint-2 checkpoint-3 checkpoint-4 checkpoint-5 checkpoint-6 checkpoint-7 checkpoint-8 checkpoint-9 last model_best model_best_adv)
CKPTS_LR25E6=(checkpoint-0 checkpoint-1 checkpoint-2 checkpoint-3 checkpoint-4 checkpoint-5 checkpoint-6 checkpoint-7 checkpoint-8 checkpoint-9 last model_best model_best_adv)

# Flatten into (model_dir, checkpoint) pairs, 13 per model, 39 total.
PAIRS=()
for ckpt in "${CKPTS_LR1E4[@]}"; do PAIRS+=("${MODEL_DIRS[0]}:${ckpt}"); done
for ckpt in "${CKPTS_LR5E5[@]}"; do PAIRS+=("${MODEL_DIRS[1]}:${ckpt}"); done
for ckpt in "${CKPTS_LR25E6[@]}"; do PAIRS+=("${MODEL_DIRS[2]}:${ckpt}"); done

IDX="${SLURM_ARRAY_TASK_ID}"
if (( IDX < 0 || IDX >= ${#PAIRS[@]} )); then
  echo "[ERROR] array index ${IDX} out of range for ${#PAIRS[@]} pairs" >&2
  exit 1
fi

PAIR="${PAIRS[$IDX]}"
MODEL_DIR_NAME="${PAIR%%:*}"
CKPT_STEM="${PAIR##*:}"
CKPT_PATH="${MODELS_ROOT}/${MODEL_DIR_NAME}/${CKPT_STEM}.pth.tar"

echo "array_index : ${IDX}"
echo "model_dir   : ${MODEL_DIR_NAME}"
echo "checkpoint  : ${CKPT_PATH}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] missing checkpoint: ${CKPT_PATH}" >&2
  exit 1
fi

if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] ImageNet val dir is not available on host $(hostname): ${VAL_DIR}" >&2
  exit 1
fi

python data_analysis/autoattack_extra_ckpts_linf6.py \
  --checkpoint "${CKPT_PATH}" \
  --model-dir "${MODELS_ROOT}/${MODEL_DIR_NAME}" \
  --val-dir "${VAL_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda
