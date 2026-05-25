#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${TIMING_CONDA_ENV:-${CONDA_DEFAULT_ENV:-ares}}"

TESTS=(
  tests/test_attack_contracts.py
  tests/test_attacker_steps.py
  tests/test_dataset_contracts.py
  tests/test_advtrain.py
  tests/test_final_eval.py
)

exec conda run -n "${CONDA_ENV}" python -m pytest -q "${TESTS[@]}"
