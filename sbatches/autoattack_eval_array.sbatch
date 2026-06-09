#!/bin/bash
#SBATCH --job-name=small_eps_autoattack
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=0
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/autoattack_eval/%A_%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

MODELS_ROOT=/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models
VAL_DIR=/groups/golan_neurogroup/bml_group/datasets/imagenet/val
MODEL_NAMES=(
  convnext_small_gradnorm_l2_16_init1
)
MODEL_NAME="${MODEL_NAMES[${SLURM_ARRAY_TASK_ID}]}"

if [[ ! -d "${MODELS_ROOT}" ]]; then
  echo "[ERROR] Models root is not available on host $(hostname): ${MODELS_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] ImageNet val dir is not available on host $(hostname): ${VAL_DIR}" >&2
  exit 1
fi

python data_analysis/autoattack_array_eval.py \
  --models-root "${MODELS_ROOT}" \
  --val-dir "${VAL_DIR}" \
  --model-name "${MODEL_NAME}" \
  --batch-size 128 \
  --num-batches 8 \
  --num-workers 6 \
  --seed 0 \
  --device cuda \
  --eps-inputs 1e-4,1e-3,1e-2,1e-1 \
  --run-id small_eps_autoattack_sweep_v1_raw224_seed0_8x128_linf-l2-l1_eps1e-4-1e-3-1e-2-1e-1 \
  --output-csv small_eps_autoattack_sweep_results.csv \
  --selection-json small_eps_autoattack_sweep_selection.json
