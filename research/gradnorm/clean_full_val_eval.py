#!/usr/bin/env python3
"""Evaluate clean ImageNet validation accuracy for external GradNorm models."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gradnorm_autoattack_eval import (
    DEFAULT_VAL_DIR,
    ROOT,
    build_dataset,
    checkpoint_path,
    load_manifest,
    load_model,
    select_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "model_manifest.yaml")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--external-dir", type=Path, default=ROOT / "external")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clean_full_val_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(out_dir / f"clean_full_val-{dt.datetime.now().strftime('%Y-%m-%d')}.log", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def evaluate_clean(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            _, pred = logits.topk(5, dim=1)
            total += targets.numel()
            top1_correct += pred[:, 0].eq(targets).sum().item()
            top5_correct += pred.eq(targets.view(-1, 1)).any(dim=1).sum().item()
    return {
        "num_images": total,
        "top1_acc": top1_correct / max(total, 1),
        "top5_acc": top5_correct / max(total, 1),
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
    }


def write_outputs(out_dir: Path, row: dict) -> None:
    json_path = out_dir / "clean_full_val_results.json"
    csv_path = out_dir / "clean_full_val_results.csv"
    with json_path.open("w") as handle:
        json.dump(row, handle, indent=2)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    models = load_manifest(args.manifest)
    _, model_spec = select_model(models, args)
    out_dir = args.results_dir / model_spec["name"] / "clean_full_val"
    logger = setup_logger(out_dir)

    if not args.val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet val dir not found: {args.val_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    logger.info("Loading %s", model_spec["name"])
    model, builder, state_key = load_model(model_spec, args, device)
    dataset = build_dataset(args.val_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )
    logger.info("Evaluating %d validation images", len(dataset))
    metrics = evaluate_clean(model, loader, device)
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_spec["name"],
        "display_name": model_spec["display_name"],
        "checkpoint_path": str(checkpoint_path(model_spec, args.models_dir)),
        "builder": builder,
        "state_key": state_key,
        "val_dir": str(args.val_dir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        **metrics,
    }
    write_outputs(out_dir, row)
    logger.info("top1=%.4f top5=%.4f num_images=%d", row["top1_acc"], row["top5_acc"], row["num_images"])
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
