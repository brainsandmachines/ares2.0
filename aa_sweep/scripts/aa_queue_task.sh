#!/bin/bash
# Body of one aa_sweep queue array task. Sourced by sbatches/aa_sweep_queue_{main,rtx6000}.sbatch;
# one task = one GPU = one (model, checkpoint kind) unit:
#
#   GPU check -> claim one unit (stock python3, no conda) -> engine -> finish (CSV census decides)
#
# Nothing to claim: cancel this array's remaining PENDING tasks and exit 0, so an empty queue
# leaves nothing running. The nightly `aa_sweep.submit` feed resubmits an array once there is work.
#
# On the time limit (--signal=B:TERM@120) or scancel: kill the engine FIRST -- never two writers on
# one CSV -- then release the unit (no attempt counted; the engine flushed every finished cell, so
# the next claimant resumes exactly there).

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ashtomer/projects/ares}"
VAL_DIR="${VAL_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/val}"

# 32 x 32 = 1024 images with the reused autoattack_sweep_selection.json: the same images as every
# older sweep row, so new cells are directly comparable. See sbatches/aa_sweep_completion.sbatch.
AA_BATCH_SIZE="${AA_BATCH_SIZE:-32}"
AA_NUM_BATCHES="${AA_NUM_BATCHES:-32}"
AA_NUM_WORKERS="${AA_NUM_WORKERS:-6}"
AA_SEED="${AA_SEED:-0}"
AA_NORMS="${AA_NORMS:-linf,l2,l1}"
AA_EPS_INPUTS="${AA_EPS_INPUTS:-1,2,4,6,8}"
MIN_GPU_MB="${AA_MIN_GPU_MB:-20000}"
LOG_PATH="${REPO_ROOT}/outs/aa_sweep/${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-x}.out"

cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

queue() { python3 -m aa_sweep.cluster_queue "$@"; }

echo "[aa_queue] task=${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} job=${SLURM_JOB_ID:-?} part=${SLURM_JOB_PARTITION:-?} host=$(hostname)"

# Before claiming, so a bad node costs no unit: bail in seconds rather than OOM hours in.
GPU_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
if [[ -z "${GPU_TOTAL}" || "${GPU_TOTAL}" -lt "${MIN_GPU_MB}" ]]; then
  echo "[ERROR] GPU on $(hostname) has ${GPU_TOTAL:-unknown} MB < ${MIN_GPU_MB} MB required" >&2
  exit 1
fi
if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] ImageNet val dir is not available on host $(hostname): ${VAL_DIR}" >&2
  exit 1
fi

line="$(queue claim --log "${LOG_PATH}")" || { echo "[ERROR] claim failed" >&2; exit 1; }
if [[ -z "${line}" ]]; then
  echo "[aa_queue] nothing pending: cancelling this array's remaining pending tasks"
  [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] && scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}" || true
  exit 0
fi
IFS=$'\t' read -r UNIT_ID UNIT_KIND UNIT_DIR <<<"${line}"
echo "[aa_queue] claimed unit=${UNIT_ID} kind=${UNIT_KIND} dir=${UNIT_DIR} gpu=${GPU_TOTAL}MB"

CHILD=""
release_and_exit() {
  if [[ -n "${CHILD}" ]]; then
    kill -TERM "${CHILD}" 2>/dev/null
    wait "${CHILD}" 2>/dev/null
  fi
  queue release "${UNIT_ID}" || true
  exit 143
}
trap release_and_exit TERM

# `main` spans 24GB and 96GB cards; tomer_advtrain works on all of them.
module load anaconda
source activate tomer_advtrain

# No --force: the engine diffs the CSV's (norm, eps) rows against the grid and attacks only what
# is missing, so it resumes a released unit and reuses the eps_norm row from training.
python data_analysis/autoattack_array_eval.py \
  --model-dir "${UNIT_DIR}" \
  --checkpoint-kinds "${UNIT_KIND}" \
  --val-dir "${VAL_DIR}" \
  --norms "${AA_NORMS}" \
  --eps-inputs "${AA_EPS_INPUTS}" \
  --batch-size "${AA_BATCH_SIZE}" \
  --num-batches "${AA_NUM_BATCHES}" \
  --num-workers "${AA_NUM_WORKERS}" \
  --seed "${AA_SEED}" \
  --device cuda \
  --plot-comparison &
CHILD=$!
wait "${CHILD}"
rc=$?
CHILD=""

queue finish "${UNIT_ID}" "${rc}"
exit "${rc}"
