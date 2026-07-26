#!/bin/bash
#SBATCH --job-name=aa_cnb_linf6_ckpts
#SBATCH --time=1-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=0-29%6
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/autoattack_extra_checkpoints/cnb_linf6/%A_%a.out

# AutoAttack, Linf norm, eps_input=6 (eval eps=6/255), 64 images in a single
# batch, over the 10 rolling checkpoint-N.pth.tar files kept in each of the
# 3 convnext_base linf cont4to6 lr-grid model dirs (excludes last/model_best/
# model_best_adv, which are already covered by other sweeps).

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

MODELS_ROOT=/home/ashtomer/projects/ares/results/models
VAL_DIR=/groups/golan_neurogroup/bml_group/datasets/imagenet/val
BATCH_SIZE=64

MODEL_DIRS=(
  "convnext_base_linf_cont4to6_lr1e4_init0"
  "convnext_base_linf_cont4to6_lr5e5_init0"
  "convnext_base_linf_cont4to6_lr25e6_init0"
)

CKPTS_LR1E4=(checkpoint-26 checkpoint-27 checkpoint-28 checkpoint-29 checkpoint-31 checkpoint-33 checkpoint-34 checkpoint-35 checkpoint-36 checkpoint-37)
CKPTS_LR5E5=(checkpoint-1 checkpoint-31 checkpoint-32 checkpoint-33 checkpoint-34 checkpoint-35 checkpoint-36 checkpoint-37 checkpoint-38 checkpoint-39)
CKPTS_LR25E6=(checkpoint-0 checkpoint-1 checkpoint-2 checkpoint-3 checkpoint-4 checkpoint-35 checkpoint-36 checkpoint-37 checkpoint-38 checkpoint-39)

# Flatten into (model_dir, checkpoint) pairs, 10 per model, 30 total.
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
