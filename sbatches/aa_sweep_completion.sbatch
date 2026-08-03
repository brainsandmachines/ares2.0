#!/bin/bash
# Complete the AutoAttack full sweep (linf/l2/l1 x eps 1,2,4,6,8) for ONE checkpoint kind of ONE
# finished model, on the shared `main` queue. Submitted one job per (model, checkpoint kind) by
# the Botero-side daily driver, `python -m aa_sweep.submit`.
#
# Inputs come in through --export (the driver sets them; both are required):
#   AA_MODEL_DIR       absolute path of the run directory, e.g.
#                      /home/ashtomer/projects/ares/results/models/convnext_base_linftrades_2_init0
#   AA_CHECKPOINT_KIND one of best | last | advbest
#
# Manual submission:
#   sbatch --job-name=aaswp_<model>_<kind> \
#     --export=ALL,AA_MODEL_DIR=<dir>,AA_CHECKPOINT_KIND=<kind> \
#     sbatches/aa_sweep_completion.sbatch
#
# The three kinds write three DIFFERENT CSVs (autoattack_sweep_results{,_last,_advbest}.csv), so
# the three jobs of a model can run concurrently without racing on a file.
#
#SBATCH --job-name=aaswp
#SBATCH --time=7-00:00:00
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint="rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000"
#SBATCH --exclude=cs-pheno-[01-12],ise-pheno-[01-12]
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/aa_sweep/%j.out

# No --mem: that is host RAM, not VRAM. The --constraint is what guarantees a >=24GB GPU (it
# excludes the 11GB gtx_1080/rtx_2080 nodes on `main`); the partition default RAM is plenty for a
# batch-size-32 eval.
#
# --exclude: the *-pheno-* nodes hold ONE rtx_3090 *and* ONE gtx_1080
# (gres=gpu:rtx_3090:1,gpu:gtx_1080:1). They satisfy the constraint on their feature list, but
# --gpus=1 does not pin which of the two cards you get, so a job there can land on the 11GB card.
# Slurm cannot OR GPU types in a --gpus request, so excluding the mixed nodes is the only
# deterministic fix. ~90 single-type nodes on `main` still qualify.

set -euo pipefail

nvidia-smi

REPO_ROOT="${REPO_ROOT:-/home/ashtomer/projects/ares}"
VAL_DIR="${VAL_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/val}"

# 32 x 32 = 1024 images -- the same count as the older 128 x 8 sweeps. Combined with the reused
# autoattack_sweep_selection.json this means the *same* 1024 images, so new rows are directly
# comparable to the eps_norm rows already sitting in the CSV.
AA_BATCH_SIZE="${AA_BATCH_SIZE:-32}"
AA_NUM_BATCHES="${AA_NUM_BATCHES:-32}"
AA_NUM_WORKERS="${AA_NUM_WORKERS:-6}"
AA_SEED="${AA_SEED:-0}"
AA_NORMS="${AA_NORMS:-linf,l2,l1}"
AA_EPS_INPUTS="${AA_EPS_INPUTS:-1,2,4,6,8}"

if [[ -z "${AA_MODEL_DIR:-}" || -z "${AA_CHECKPOINT_KIND:-}" ]]; then
  echo "[ERROR] AA_MODEL_DIR and AA_CHECKPOINT_KIND must be exported (see header)" >&2
  exit 2
fi

if [[ ! -d "${AA_MODEL_DIR}" ]]; then
  echo "[ERROR] Model dir is not available on host $(hostname): ${AA_MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] ImageNet val dir is not available on host $(hostname): ${VAL_DIR}" >&2
  exit 1
fi

# Defence in depth behind --constraint/--exclude: bail in seconds rather than OOM twenty hours
# into a sweep if this job somehow landed on an 11GB card.
GPU_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
MIN_GPU_MB="${AA_MIN_GPU_MB:-20000}"
if [[ -z "${GPU_TOTAL}" || "${GPU_TOTAL}" -lt "${MIN_GPU_MB}" ]]; then
  echo "[ERROR] GPU on $(hostname) has ${GPU_TOTAL:-unknown} MB < ${MIN_GPU_MB} MB required for bsz ${AA_BATCH_SIZE}" >&2
  exit 1
fi
echo "[INFO] gpu_total_mb   : ${GPU_TOTAL}"

mkdir -p "${REPO_ROOT}/outs/aa_sweep"

# `main` spans 24GB and 96GB cards; tomer_advtrain works on all of them.
module load anaconda
source activate tomer_advtrain

cd "${REPO_ROOT}"

MODEL_NAME="$(basename "${AA_MODEL_DIR}")"
echo "[INFO] model_dir      : ${AA_MODEL_DIR}"
echo "[INFO] checkpoint_kind: ${AA_CHECKPOINT_KIND}"
echo "[INFO] grid           : ${AA_NORMS} x ${AA_EPS_INPUTS}"
echo "[INFO] images         : ${AA_BATCH_SIZE} x ${AA_NUM_BATCHES}"

# Make squeue readable, and let the driver's dedupe check see this (model, kind) as in flight.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" \
    JobName="aaswp_${MODEL_NAME}_${AA_CHECKPOINT_KIND}" || true
fi

# No --force: the engine diffs the CSV's existing (norm, eps) rows against the grid and attacks
# only what is missing, so an eps_norm result already on disk is reused, never recomputed.
python data_analysis/autoattack_array_eval.py \
  --model-dir "${AA_MODEL_DIR}" \
  --checkpoint-kinds "${AA_CHECKPOINT_KIND}" \
  --val-dir "${VAL_DIR}" \
  --norms "${AA_NORMS}" \
  --eps-inputs "${AA_EPS_INPUTS}" \
  --batch-size "${AA_BATCH_SIZE}" \
  --num-batches "${AA_NUM_BATCHES}" \
  --num-workers "${AA_NUM_WORKERS}" \
  --seed "${AA_SEED}" \
  --device cuda \
  --plot-comparison
