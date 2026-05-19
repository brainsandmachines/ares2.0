#!/bin/bash
#SBATCH --job-name=autoattack_eval
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/autoattack_eval/%j.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

MODELS_ROOT=/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models
VAL_DIR=/groups/golan_neurogroup/bml_group/datasets/imagenet/val

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
  --model-name convnext_small_linftrades_8_init2 \
  --batch-size 128 \
  --num-batches 8 \
  --num-workers 6 \
  --seed 0 \
  --device cuda \
  --force
