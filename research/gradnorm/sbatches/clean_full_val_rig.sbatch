#!/bin/bash
#SBATCH --job-name=rig_clean_full
#SBATCH --time=08:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/research/gradnorm/logs/slurm/%A.clean_full_val.out

set -euo pipefail

module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

mkdir -p research/gradnorm/logs/slurm
nvidia-smi

VAL_DIR="${VAL_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/val}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] ImageNet val dir is not available on host $(hostname): ${VAL_DIR}" >&2
  exit 1
fi

python research/gradnorm/clean_full_val_eval.py \
  --model-name rig_gradnorm_resnet50_gelu \
  --val-dir "${VAL_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda
