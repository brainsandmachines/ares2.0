#!/bin/bash
#SBATCH --job-name=aa_eps6_12_backfill
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=0-106%4
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/autoattack_eval/%A/%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain_pro

cd /home/ashtomer/projects/ares

MODELS_ROOT=/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models
VAL_DIR=/groups/golan_neurogroup/bml_group/datasets/imagenet/val
AA_EPS_INPUTS="${AA_EPS_INPUTS:-6,12}"
AA_OUTPUT_CSV="${AA_OUTPUT_CSV:-autoattack_sweep_results.csv}"
AA_SELECTION_JSON="${AA_SELECTION_JSON:-autoattack_sweep_selection.json}"
AA_RUN_ID="${AA_RUN_ID:-autoattack_sweep_v1_raw224_seed0_8x128_linf-l2-l1_eps6-12_backfill}"
AA_EXISTING_CSV_ONLY="${AA_EXISTING_CSV_ONLY:-1}"
AA_BATCH_SIZE="${AA_BATCH_SIZE:-128}"
AA_NUM_BATCHES="${AA_NUM_BATCHES:-8}"
AA_NUM_WORKERS="${AA_NUM_WORKERS:-6}"
AA_SEED="${AA_SEED:-0}"

if [[ ! -d "${MODELS_ROOT}" ]]; then
  echo "[ERROR] Models root is not available on host $(hostname): ${MODELS_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] ImageNet val dir is not available on host $(hostname): ${VAL_DIR}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${AA_EXISTING_CSV_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--existing-csv-only)
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
  "${EXTRA_ARGS[@]}"
