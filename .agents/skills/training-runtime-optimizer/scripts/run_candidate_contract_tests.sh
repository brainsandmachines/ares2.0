#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/.agents/skills/training-runtime-optimizer/scripts"

CONDA_ENV="${TIMING_CONDA_ENV:-${CONDA_DEFAULT_ENV:-ares}}"

exec conda run -n "${CONDA_ENV}" python "${SCRIPT_DIR}/run_candidate_contract_tests.py" "$@"
