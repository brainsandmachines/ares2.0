#!/bin/bash
#SBATCH --job-name=gradnorm_ext_aa
#SBATCH --time=7-00:00:00
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/research/gradnorm/logs/slurm/%A_%a.out
#SBATCH --array=0

set -euo pipefail

module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

mkdir -p research/gradnorm/logs/slurm
nvidia-smi

VAL_DIR="${VAL_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/val}"

if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] ImageNet val dir is not available on host $(hostname): ${VAL_DIR}" >&2
  exit 1
fi

python research/gradnorm/gradnorm_autoattack_eval.py \
  --model-index "${SLURM_ARRAY_TASK_ID}" \
  --val-dir "${VAL_DIR}" \
  --batch-size 16 \
  --num-batches 16 \
  --num-workers 6 \
  --seed 0 \
  --device cuda \
  --norms linf,l2 \
  --eps-inputs 0.1,0.5,1,2,4,6,8
