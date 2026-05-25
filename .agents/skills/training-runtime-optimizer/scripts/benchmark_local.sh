#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/.agents/skills/training-runtime-optimizer/scripts"

protocol=""
candidate="baseline"
output_dir=""
train_dir=""
eval_dir=""
batch_sizes="128,112,96"
warmup_iters="1"
measured_iters="5"
num_workers="6"
seed="0"
conda_env="${TIMING_CONDA_ENV:-${CONDA_DEFAULT_ENV:-ares}}"
artifact_mode="compact"

usage() {
  cat <<'EOF'
Usage:
  benchmark_local.sh --protocol <name> --output-dir <dir> [options]

Options:
  --candidate <name>
  --train-dir <path>
  --eval-dir <path>
  --batch-sizes <csv>
  --warmup-iters <n>
  --measured-iters <n>
  --num-workers <n>
  --seed <n>
  --conda-env <name>
  --artifact-mode <compact|keep-all>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --protocol) protocol="$2"; shift 2 ;;
    --candidate) candidate="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --train-dir) train_dir="$2"; shift 2 ;;
    --eval-dir) eval_dir="$2"; shift 2 ;;
    --batch-sizes) batch_sizes="$2"; shift 2 ;;
    --warmup-iters) warmup_iters="$2"; shift 2 ;;
    --measured-iters) measured_iters="$2"; shift 2 ;;
    --num-workers) num_workers="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --conda-env) conda_env="$2"; shift 2 ;;
    --artifact-mode) artifact_mode="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${protocol}" || -z "${output_dir}" ]]; then
  echo "--protocol and --output-dir are required" >&2
  exit 2
fi

if [[ "${artifact_mode}" != "compact" && "${artifact_mode}" != "keep-all" ]]; then
  echo "--artifact-mode must be compact or keep-all" >&2
  exit 2
fi

if [[ -z "${train_dir}" || -z "${eval_dir}" ]]; then
  candidates=(
    "/mnt/data/datasets/imagenet_sample/train|/mnt/data/datasets/imagenet_sample/val"
    "/storage/test/bml_group/tomerash/datasets/imagenet/train|/storage/test/bml_group/tomerash/datasets/imagenet/val"
    "${HOME}/datasets/imagenet/train|${HOME}/datasets/imagenet/val"
    "/mnt/data/datasets/imagenet/train|/mnt/data/datasets/imagenet/val"
  )
  for pair in "${candidates[@]}"; do
    train_candidate="${pair%%|*}"
    eval_candidate="${pair##*|}"
    if [[ -d "${train_candidate}" && -d "${eval_candidate}" ]]; then
      train_dir="${train_candidate}"
      eval_dir="${eval_candidate}"
      break
    fi
  done
fi

if [[ -z "${train_dir}" || -z "${eval_dir}" ]]; then
  echo "Could not resolve dataset roots" >&2
  exit 1
fi

mkdir -p "${output_dir}"
printf '{"artifact_mode":"%s","policy":"compact summaries are default; keep full logs only for winners, failures, or requested audits"}\n' "${artifact_mode}" > "${output_dir}/artifact_policy.json"

export TIMING_CONDA_ENV="${conda_env}"

bash "${SCRIPT_DIR}/run_contract_tests.sh"

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

summary_csv="${output_dir}/summary.csv"
printf "protocol,candidate,batch_size,status,result_json\n" > "${summary_csv}"

IFS=',' read -r -a bsz_values <<< "${batch_sizes}"
for bsz in "${bsz_values[@]}"; do
  run_dir="${output_dir}/${protocol}/${candidate}/bsz_${bsz}"
  mkdir -p "${run_dir}"
  result_json="${run_dir}/result.json"

  if conda run -n "${conda_env}" python "${SCRIPT_DIR}/run_candidate_benchmark.py" \
      --protocol "${protocol}" \
      --candidate "${candidate}" \
      --train-dir "${train_dir}" \
      --eval-dir "${eval_dir}" \
      --output-dir "${run_dir}" \
      --batch-size "${bsz}" \
      --warmup-iters "${warmup_iters}" \
      --measured-iters "${measured_iters}" \
      --num-workers "${num_workers}" \
      --seed "${seed}"; then
    status="ok"
  else
    status="failed"
  fi

  printf "%s,%s,%s,%s,%s\n" "${protocol}" "${candidate}" "${bsz}" "${status}" "${result_json}" >> "${summary_csv}"

  if [[ "${status}" == "ok" ]]; then
    break
  fi

  if [[ -f "${result_json}" ]] && rg -q 'OutOfMemoryError|CUDA out of memory' "${result_json}"; then
    continue
  fi

  break
done

echo "Wrote ${summary_csv}"
