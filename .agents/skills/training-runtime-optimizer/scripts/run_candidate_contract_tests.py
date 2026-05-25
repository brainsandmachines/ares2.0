#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import robust_training.adversarial_training as advt

from run_candidate_benchmark import (
    EXAMPLE_HARNESS_BEHAVIORS,
    TimerCollector,
    apply_backend_candidate,
    apply_candidate_args,
    compose_args,
    patch_attack_flatten_methods,
    patch_build_model,
    patch_create_optimizer,
    patch_train_one_epoch,
    restore_attack_flatten_methods,
    restore_backend_candidate,
)


TESTS = [
    "tests/test_attack_contracts.py",
    "tests/test_attacker_steps.py",
    "tests/test_dataset_contracts.py",
    "tests/test_advtrain.py",
    "tests/test_final_eval.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contract tests with timing candidate hooks applied")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--train-dir", default="/tmp/train")
    parser.add_argument("--eval-dir", default="/tmp/val")
    parser.add_argument("--output-dir", default="/tmp/ares-candidate-contract")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--measured-iters", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    cli_args = parse_args()
    if cli_args.candidate not in EXAMPLE_HARNESS_BEHAVIORS:
        print(
            f"warning: candidate '{cli_args.candidate}' has no built-in harness behavior; "
            "running tests with baseline behavior unless custom hooks were added",
            file=sys.stderr,
        )

    composed_args = compose_args(cli_args.protocol, cli_args)
    apply_candidate_args(composed_args, cli_args.candidate)

    previous_backend = apply_backend_candidate(cli_args.candidate)
    attack_flatten_originals = patch_attack_flatten_methods(cli_args.candidate)
    original_build_model = advt.build_model
    original_create_optimizer = advt.create_optimizer_v2
    original_train_one_epoch = advt.train_one_epoch

    # Apply candidate hooks that the timing harness uses. Tests may monkeypatch
    # these functions themselves; pytest's monkeypatch fixture will then layer on
    # top of this candidate-applied baseline and restore safely.
    timer = TimerCollector(warmup_iters=0, measured_iters=1)
    advt.build_model = patch_build_model(cli_args.candidate)
    advt.create_optimizer_v2 = patch_create_optimizer(cli_args.candidate)
    advt.train_one_epoch = patch_train_one_epoch(timer, cli_args.candidate)

    try:
        return pytest.main(["-q", *TESTS])
    finally:
        advt.build_model = original_build_model
        advt.create_optimizer_v2 = original_create_optimizer
        advt.train_one_epoch = original_train_one_epoch
        restore_attack_flatten_methods(attack_flatten_originals)
        restore_backend_candidate(previous_backend)


if __name__ == "__main__":
    raise SystemExit(main())
