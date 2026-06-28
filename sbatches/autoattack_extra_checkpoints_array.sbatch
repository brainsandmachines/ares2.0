#!/bin/bash
#SBATCH --job-name=aa_extra_ckpts
#SBATCH --time=7-00:00:00
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --array=0-199%8
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/autoattack_extra_checkpoints/%A/%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

MODELS_ROOT="${MODELS_ROOT:-/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models}"
VAL_DIR="${VAL_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/val}"
AA_EPS_INPUTS="${AA_EPS_INPUTS:-1,2,4,6,8,12}"
AA_OUTPUT_CSV="${AA_OUTPUT_CSV:-autoattack_sweep_results.csv}"
AA_SELECTION_JSON="${AA_SELECTION_JSON:-autoattack_sweep_selection.json}"
AA_RUN_ID="${AA_RUN_ID:-autoattack_sweep_v1_raw224_seed0_8x16_linf-l2-l1_eps1-2-4-6-8-12_extra_ckpts}"
AA_BATCH_SIZE="${AA_BATCH_SIZE:-32}"
AA_NUM_BATCHES="${AA_NUM_BATCHES:-32}"
AA_NUM_WORKERS="${AA_NUM_WORKERS:-6}"
AA_SEED="${AA_SEED:-0}"
AA_CHECKPOINT_KINDS="${AA_CHECKPOINT_KINDS:-last,advbest}"

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
  --batch-size "${AA_BATCH_SIZE}" \
  --num-batches "${AA_NUM_BATCHES}" \
  --num-workers "${AA_NUM_WORKERS}" \
  --seed "${AA_SEED}" \
  --device cuda \
  --eps-inputs "${AA_EPS_INPUTS}" \
  --run-id "${AA_RUN_ID}" \
  --output-csv "${AA_OUTPUT_CSV}" \
  --selection-json "${AA_SELECTION_JSON}" \
  --checkpoint-kinds "${AA_CHECKPOINT_KINDS}" \
  --existing-csv-only
