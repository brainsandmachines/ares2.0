#!/usr/bin/env python3
"""One-off AutoAttack Linf eps=6 sweep over specific numbered checkpoints.

Evaluates a single checkpoint file (not the best/last/advbest kinds handled by
autoattack_array_eval.py) against AutoAttack, Linf norm, eps_input=6 (eval eps
= 6/255), using exactly one batch of --batch-size images. Appends a single row
to a per-checkpoint output CSV so parallel Slurm array tasks never race on the
same file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.autoattack_array_eval import (  # noqa: E402
    build_selected_loader,
    clean_accuracy,
    eps_eval,
    extract_epoch_from_checkpoint,
    load_model,
    run_autoattack_setting,
    save_selection,
    set_seed,
    setup_logger,
    validate_raw_batch,
)
from types import SimpleNamespace  # noqa: E402

NORM = "linf"
EPS_INPUT = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-csv", type=Path, default=None,
                         help="Defaults to <model-dir>/autoattack_extra_ckpts_linf6_<checkpoint-stem>.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    model_dir = args.model_dir
    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    ckpt_stem = checkpoint_path.name
    for suffix in (".pth.tar", ".pth"):
        if ckpt_stem.endswith(suffix):
            ckpt_stem = ckpt_stem[: -len(suffix)]
            break

    output_csv = args.output_csv or (model_dir / f"autoattack_extra_ckpts_linf6_{ckpt_stem}.csv")
    logger = setup_logger(model_dir, dry_run=False)

    set_seed(args.seed)
    device = torch.device(args.device)
    model, eval_cfg, state_key = load_model(checkpoint_path, device)

    loader, selected_indices, selected_targets = build_selected_loader(
        args.val_dir,
        eval_cfg,
        batch_size=args.batch_size,
        num_batches=1,
        num_workers=6,
        seed=args.seed,
    )
    validate_raw_batch(loader)

    selection_args = SimpleNamespace(seed=args.seed, batch_size=args.batch_size, num_batches=1)
    selection_path = save_selection(
        model_dir,
        selected_indices,
        selected_targets,
        selection_args,
        run_id=f"autoattack_extra_ckpts_linf6_{ckpt_stem}",
        selection_json=f"autoattack_extra_ckpts_linf6_{ckpt_stem}_selection.json",
    )

    logger.info(
        "checkpoint=%s selected %d images from %d unique classes",
        checkpoint_path.name, len(selected_indices), len(set(selected_targets)),
    )

    clean_acc = clean_accuracy(model, loader, device)
    logger.info("clean_acc=%.4f", clean_acc * 100.0)

    epsilon = eps_eval(NORM, EPS_INPUT)
    set_seed(args.seed)
    logger.info("running AutoAttack norm=%s epsilon_input=%s epsilon_eval=%s", NORM, EPS_INPUT, epsilon)
    robust_acc = run_autoattack_setting(model, loader, device, NORM, epsilon, args.seed, logger)

    epoch = extract_epoch_from_checkpoint(checkpoint_path)
    row = {
        "run_id": f"autoattack_extra_ckpts_linf6_1x{args.batch_size}",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_dir.name,
        "checkpoint_kind": ckpt_stem,
        "checkpoint_path": str(checkpoint_path),
        "state_dict_used": state_key,
        "epoch": epoch,
        "attack_norm": NORM,
        "epsilon_input": EPS_INPUT,
        "epsilon_eval": epsilon,
        "clean_acc": clean_acc * 100.0,
        "robust_acc": robust_acc * 100.0,
        "num_images": len(selected_indices),
        "batch_size": args.batch_size,
        "num_batches": 1,
        "seed": args.seed,
        "selection_json": str(selection_path),
    }

    write_header = not output_csv.exists()
    with output_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("wrote row to %s", output_csv)


if __name__ == "__main__":
    main()
