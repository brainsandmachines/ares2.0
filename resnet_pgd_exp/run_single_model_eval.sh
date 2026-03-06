#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_ROOT="${MODELS_ROOT:-/storage/test/bml_group/tomerash/madry_orig_robustmodels}"
VAL_DIR="${VAL_DIR:-/storage/test/bml_group/tomerash/datasets/imagenet/val}"
OUT_ROOT="${OUT_ROOT:-${MODELS_ROOT}/pgd_eval_resnet50}"
LOCAL_OUT_ROOT="${LOCAL_OUT_ROOT:-${REPO_ROOT}/resnet_pgd_exp/results}"
INPUT_MODE="${INPUT_MODE:-auto}"
PGD_EPS="${PGD_EPS:-0,0.01,0.03,0.05,0.1,0.25,0.5,1,3,5}"
PGD_NORMS="${PGD_NORMS:-linf,l2,l1}"
PGD_ATTACK_STEPS="${PGD_ATTACK_STEPS:-10}"
PGD_BATCH_SIZE="${PGD_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-0}}"

if [[ ! -d "${MODELS_ROOT}" ]]; then
  echo "[ERROR] MODELS_ROOT does not exist: ${MODELS_ROOT}"
  exit 1
fi
if [[ ! -d "${VAL_DIR}" ]]; then
  echo "[ERROR] VAL_DIR does not exist: ${VAL_DIR}"
  exit 1
fi

mapfile -t MODELS < <(
  find "${MODELS_ROOT}" -maxdepth 1 -type f -name 'resnet50_l2_eps*.ckpt' \
    ! -iname '*wide*' | sort -V
)

NUM_MODELS="${#MODELS[@]}"
echo "[INFO] discovered_resnet50_models=${NUM_MODELS}"

if [[ "${NUM_MODELS}" -eq 0 ]]; then
  echo "[ERROR] No matching checkpoints found under ${MODELS_ROOT}"
  exit 1
fi

if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${NUM_MODELS}" ]]; then
  echo "[INFO] TASK_ID=${TASK_ID} out of range [0,$((NUM_MODELS - 1))]. Exiting."
  exit 0
fi

CKPT="${MODELS[$TASK_ID]}"
MODEL_STEM="$(basename "${CKPT}" .ckpt)"
OUT_DIR="${OUT_ROOT}/${MODEL_STEM}"
LOCAL_OUT_DIR="${LOCAL_OUT_ROOT}/${MODEL_STEM}"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOCAL_OUT_DIR}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="PGD_madry_${MODEL_STEM}" || true
fi

echo "[INFO] task_id=${TASK_ID}"
echo "[INFO] checkpoint=${CKPT}"
echo "[INFO] model_stem=${MODEL_STEM}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] local_out_dir=${LOCAL_OUT_DIR}"
echo "[INFO] val_dir=${VAL_DIR}"
echo "[INFO] input_mode=${INPUT_MODE}"

INPUT_MODE_EFFECTIVE="${INPUT_MODE}"
if [[ "${INPUT_MODE}" == "auto" ]]; then
  if command -v rg >/dev/null 2>&1; then
    HAS_NORMALIZER_SIG=0
    if strings -n 8 "${CKPT}" | rg -qi 'normalizer|attacker\.normalize|module\.normalizer'; then
      HAS_NORMALIZER_SIG=1
    fi
  else
    HAS_NORMALIZER_SIG=0
    if strings -n 8 "${CKPT}" | grep -Eqi 'normalizer|attacker\.normalize|module\.normalizer'; then
      HAS_NORMALIZER_SIG=1
    fi
  fi

  if [[ "${HAS_NORMALIZER_SIG}" -eq 1 ]]; then
    INPUT_MODE_EFFECTIVE="raw"
    echo "[INFO] detected normalizer signature in checkpoint; forcing raw [0,1] loader and skipping probe"
  fi
fi
echo "[INFO] input_mode_effective=${INPUT_MODE_EFFECTIVE}"

cd "${REPO_ROOT}"
python resnet_pgd_exp/madry_resnet50_pgd_eval.py \
  --checkpoint "${CKPT}" \
  --val-dir "${VAL_DIR}" \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --input-mode "${INPUT_MODE_EFFECTIVE}" \
  --pgd-eps "${PGD_EPS}" \
  --pgd-norms "${PGD_NORMS}" \
  --pgd-attack-steps "${PGD_ATTACK_STEPS}" \
  --pgd-batch-size "${PGD_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

if [[ -f "${OUT_DIR}/pgd_validation_results.csv" ]]; then
  cp "${OUT_DIR}/pgd_validation_results.csv" "${LOCAL_OUT_DIR}/pgd_validation_results.csv"
fi

LATEST_LOG="$(ls -1t "${OUT_DIR}"/madry_resnet50_pgd_eval-*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_LOG}" && -f "${LATEST_LOG}" ]]; then
  cp "${LATEST_LOG}" "${LOCAL_OUT_DIR}/$(basename "${LATEST_LOG}")"
fi

echo "[INFO] mirrored_outputs_to=${LOCAL_OUT_DIR}"
