#!/usr/bin/env python3
"""Run a deterministic AutoAttack sweep for one Slurm array-selected model."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import torch
from autoattack import AutoAttack
from timm.models import create_model
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODELS_ROOT = Path("/home/ashtomer/projects/ares/results/models")
DEFAULT_VAL_DIR = Path("/groups/golan_neurogroup/bml_group/datasets/imagenet/val")
OUTPUT_CSV = "autoattack_sweep_results.csv"
SELECTION_JSON = "autoattack_sweep_selection.json"
RUN_ID = "autoattack_sweep_v1_raw224_seed0_8x16_linf-l2-l1_eps1-2-4-8-16"
EPOCH_KEYS = ("epoch", "train_epoch", "last_epoch", "current_epoch")
CKPT_CANDIDATES = (
    "model_best.pth.tar",
    "model_best.pth",
    "last.pth.model",
    "last.pth.tar",
    "last.pth",
)
NORMS = ("linf", "l2", "l1")
EPS_INPUTS = (1.0, 2.0, 4.0, 8.0, 16.0)
LINF_DIVISOR = 255.0
L1_MULTIPLIER = 255.0 / 2.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean: Iterable[float], std: Iterable[float]):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(list(mean)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(list(std)).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slurm-array AutoAttack sweep for one model.")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--array-index", type=int, default=None, help="Defaults to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--model-name", default=None, help="Evaluate this model directory name instead of selecting by array index.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--force", action="store_true", help="Rerun even if this sweep CSV is complete.")
    parser.add_argument("--dry-run", action="store_true", help="Print selection/model info without running attacks.")
    parser.add_argument("--list-models", action="store_true", help="List eligible models and exit.")
    parser.add_argument("--max-settings", type=int, default=None, help="Debug: run only the first N settings.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(model_dir: Path, dry_run: bool) -> logging.Logger:
    logger = logging.getLogger("autoattack_array_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if not dry_run:
        log_path = model_dir / f"autoattack_sweep-{dt.datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def model_dirs(root: Path) -> list[Path]:
    out = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith(".") or path.name.startswith("__"):
            continue
        low = path.name.lower()
        if path.name == "old_models" or "old" in low:
            continue
        out.append(path)
    return out


def find_checkpoint(model_dir: Path) -> Optional[Path]:
    for name in CKPT_CANDIDATES:
        path = model_dir / name
        if path.exists():
            return path
    return None


def extract_epoch_from_checkpoint(ckpt_path: Path) -> Optional[int]:
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    for key in EPOCH_KEYS:
        value = obj.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    training_state = obj.get("training_state")
    if isinstance(training_state, dict):
        for key in EPOCH_KEYS:
            value = training_state.get(key)
            if isinstance(value, (int, float)):
                return int(value)
    return None


def extract_epoch_from_summary(model_dir: Path) -> Optional[int]:
    summary = model_dir / "summary.csv"
    if not summary.exists():
        return None
    last_epoch = None
    try:
        with summary.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                raw = row.get("epoch")
                if raw in (None, ""):
                    continue
                try:
                    last_epoch = int(float(raw))
                except Exception:
                    continue
    except Exception:
        return None
    return last_epoch


def eligible_models(root: Path) -> list[dict]:
    rows = []
    for model_dir in model_dirs(root):
        ckpt = find_checkpoint(model_dir)
        epoch = extract_epoch_from_summary(model_dir)
        if epoch is None and ckpt is not None:
            epoch = extract_epoch_from_checkpoint(ckpt)
        if ckpt is not None and epoch is not None and epoch >= 199:
            rows.append({"model_dir": model_dir, "checkpoint": ckpt, "epoch": epoch})
    return rows


def infer_eval_cfg(ckpt: dict) -> SimpleNamespace:
    ckpt_args = ckpt.get("args")
    return SimpleNamespace(
        mean=tuple(getattr(ckpt_args, "mean", IMAGENET_MEAN)),
        std=tuple(getattr(ckpt_args, "std", IMAGENET_STD)),
        input_size=int(getattr(ckpt_args, "input_size", 224)),
        interpolation=str(getattr(ckpt_args, "interpolation", "bicubic")),
        crop_pct=float(getattr(ckpt_args, "crop_pct", 0.875)),
        num_classes=int(getattr(ckpt_args, "num_classes", 1000)),
    )


def get_eval_resize(eval_cfg: SimpleNamespace) -> int:
    return max(int(round(eval_cfg.input_size / max(eval_cfg.crop_pct, 1e-6))), eval_cfg.input_size)


def strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith("module."):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def get_ckpt_arg(ckpt_args, key: str, default=None):
    if ckpt_args is None:
        return default
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(key, default)
    return getattr(ckpt_args, key, default)


def create_model_from_checkpoint(arch: str, ckpt_args, eval_cfg: SimpleNamespace) -> torch.nn.Module:
    if arch != "convnext_small_v1":
        return create_model(arch, pretrained=False, num_classes=eval_cfg.num_classes)

    from ares.model.v1_convnext import V1ConvNeXt

    return V1ConvNeXt(
        backbone_name="convnext_small",
        input_size=eval_cfg.input_size,
        num_classes=eval_cfg.num_classes,
        drop_rate=get_ckpt_arg(ckpt_args, "drop", 0.0),
        drop_path_rate=get_ckpt_arg(ckpt_args, "drop_path", 0.0),
        drop_block_rate=get_ckpt_arg(ckpt_args, "drop_block", None),
        global_pool=get_ckpt_arg(ckpt_args, "gp", None),
        bn_momentum=get_ckpt_arg(ckpt_args, "bn_momentum", None),
        bn_eps=get_ckpt_arg(ckpt_args, "bn_eps", None),
        v1_noise_train_only=get_ckpt_arg(ckpt_args, "v1_noise_train_only", True),
        visual_degrees=get_ckpt_arg(ckpt_args, "v1_visual_degrees", 8),
        stride=get_ckpt_arg(ckpt_args, "v1_stride", 4),
        ksize=get_ckpt_arg(ckpt_args, "v1_ksize", 25),
        sf_corr=get_ckpt_arg(ckpt_args, "v1_sf_corr", 0.75),
        sf_max=get_ckpt_arg(ckpt_args, "v1_sf_max", 9),
        sf_min=get_ckpt_arg(ckpt_args, "v1_sf_min", 0),
        rand_param=get_ckpt_arg(ckpt_args, "v1_rand_param", False),
        gabor_seed=get_ckpt_arg(ckpt_args, "v1_gabor_seed", 0),
        simple_channels=get_ckpt_arg(ckpt_args, "v1_simple_channels", 256),
        complex_channels=get_ckpt_arg(ckpt_args, "v1_complex_channels", 256),
        noise_mode=get_ckpt_arg(ckpt_args, "v1_noise_mode", None),
        noise_scale=get_ckpt_arg(ckpt_args, "v1_noise_scale", 0.35),
        noise_level=get_ckpt_arg(ckpt_args, "v1_noise_level", 0.07),
        k_exc=get_ckpt_arg(ckpt_args, "v1_k_exc", 25),
    )


def load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, SimpleNamespace, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    eval_cfg = infer_eval_cfg(ckpt)
    arch = ckpt.get("arch")
    if arch is None:
        raise ValueError(f"Missing arch in checkpoint: {ckpt_path}")

    model = create_model_from_checkpoint(arch, ckpt.get("args"), eval_cfg)
    state_key = "state_dict_ema" if "state_dict_ema" in ckpt else "state_dict"
    if state_key not in ckpt:
        raise ValueError(f"Missing {state_key} in checkpoint: {ckpt_path}")
    model.load_state_dict(strip_module_prefix(ckpt[state_key]), strict=True)
    model = NormalizeWrapper(model, eval_cfg.mean, eval_cfg.std).to(device).eval()
    return model, eval_cfg, state_key


def build_raw_dataset(val_dir: Path, eval_cfg: SimpleNamespace) -> datasets.ImageFolder:
    transform = transforms.Compose([
        transforms.Resize(get_eval_resize(eval_cfg)),
        transforms.CenterCrop(eval_cfg.input_size),
        transforms.ToTensor(),
    ])
    return datasets.ImageFolder(str(val_dir), transform=transform)


def select_balanced_indices(ds: datasets.ImageFolder, total_images: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for idx, (_, target) in enumerate(ds.samples):
        by_class.setdefault(int(target), []).append(idx)

    if total_images > len(ds.samples):
        raise ValueError(f"Need {total_images} images, found only {len(ds.samples)} in {ds.root}")

    rng = random.Random(seed)
    selected: list[int] = []
    classes = sorted(by_class)
    per_class = {class_idx: rng.sample(indices, len(indices)) for class_idx, indices in by_class.items()}
    positions = {class_idx: 0 for class_idx in classes}

    while len(selected) < total_images:
        available_classes = [class_idx for class_idx in classes if positions[class_idx] < len(per_class[class_idx])]
        if not available_classes:
            break
        for class_idx in rng.sample(available_classes, len(available_classes)):
            selected.append(per_class[class_idx][positions[class_idx]])
            positions[class_idx] += 1
            if len(selected) == total_images:
                break

    rng.shuffle(selected)
    return selected


def build_selected_loader(
    val_dir: Path,
    eval_cfg: SimpleNamespace,
    batch_size: int,
    num_batches: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, list[int], list[int]]:
    ds = build_raw_dataset(val_dir, eval_cfg)
    total_images = batch_size * num_batches
    selected_indices = select_balanced_indices(ds, total_images, seed)
    selected_targets = [int(ds.samples[idx][1]) for idx in selected_indices]
    loader = DataLoader(
        Subset(ds, selected_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, selected_indices, selected_targets


def eps_eval(norm: str, eps_input: float) -> float:
    if norm == "linf":
        return float(eps_input) / LINF_DIVISOR
    if norm == "l1":
        return float(eps_input) * L1_MULTIPLIER
    if norm == "l2":
        return float(eps_input)
    raise ValueError(f"Unsupported norm: {norm}")


def aa_norm(norm: str) -> str:
    return {"linf": "Linf", "l2": "L2", "l1": "L1"}[norm]


def expected_settings(max_settings: Optional[int] = None) -> list[tuple[str, float, float]]:
    settings = [(norm, eps_input, eps_eval(norm, eps_input)) for norm in NORMS for eps_input in EPS_INPUTS]
    return settings[:max_settings] if max_settings is not None else settings


def _to_abs_norm(path: str | Path) -> str:
    return os.path.normpath(os.path.abspath(str(path)))


def _checkpoint_matches(row_checkpoint_path: str | None, target_checkpoint_path: str | Path | None) -> bool:
    if target_checkpoint_path is None:
        return True
    row_raw = str(row_checkpoint_path or "").strip()
    if not row_raw:
        return False

    row_norm = os.path.normpath(row_raw)
    row_abs = _to_abs_norm(row_raw)
    target_norm = os.path.normpath(str(target_checkpoint_path))
    target_abs = _to_abs_norm(target_checkpoint_path)

    if row_abs == target_abs or row_norm == target_norm:
        return True
    if row_abs.endswith(target_norm) or target_abs.endswith(row_norm):
        return True
    return os.path.basename(row_norm) == os.path.basename(target_norm)


def is_complete_output(
    csv_path: Path,
    max_settings: Optional[int],
    checkpoint_path: str | Path | None = None,
    model_name: str | None = None,
) -> bool:
    if not csv_path.exists() or max_settings is not None:
        return False
    expected = {(norm, eps_input) for norm, eps_input, _ in expected_settings()}
    found = set()
    try:
        with csv_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("run_id") != RUN_ID:
                    return False
                if model_name is not None and row.get("model_name") != model_name:
                    continue
                if not _checkpoint_matches(row.get("checkpoint_path"), checkpoint_path):
                    continue
                found.add((row.get("attack_norm"), float(row.get("epsilon_input", "nan"))))
    except Exception:
        return False
    return found == expected


def save_selection(model_dir: Path, selected_indices: list[int], selected_targets: list[int], args: argparse.Namespace) -> None:
    payload = {
        "run_id": RUN_ID,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "num_images": len(selected_indices),
        "selected_indices": selected_indices,
        "selected_labels": selected_targets,
        "selected_targets": selected_targets,
    }
    with (model_dir / SELECTION_JSON).open("w") as handle:
        json.dump(payload, handle, indent=2)


def validate_raw_batch(loader: DataLoader) -> None:
    x, _ = next(iter(loader))
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"Expected NCHW RGB batch, got shape {tuple(x.shape)}")
    min_val = float(x.min().item())
    max_val = float(x.max().item())
    if min_val < -1e-6 or max_val > 1.0 + 1e-6:
        raise ValueError(f"AutoAttack input must be raw [0,1] pixels, got min={min_val} max={max_val}")


def clean_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def run_autoattack_setting(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm: str,
    epsilon: float,
    seed: int,
    logger: logging.Logger,
) -> float:
    adversary = AutoAttack(
        model,
        norm=aa_norm(norm),
        eps=epsilon,
        seed=seed,
        version="standard",
        verbose=True,
        device=str(device),
    )
    robust_correct, robust_total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = run_autoattack_with_fallback(adversary, x, y, logger)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        robust_correct += (pred == y).sum().item()
        robust_total += y.size(0)
    return robust_correct / max(robust_total, 1)


def run_autoattack_with_fallback(adversary: AutoAttack, x: torch.Tensor, y: torch.Tensor, logger: logging.Logger) -> torch.Tensor:
    try:
        return adversary.run_standard_evaluation(x, y, bs=x.size(0))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or x.size(0) == 1:
            raise
        logger.warning("AutoAttack OOM at batch=%d; splitting batch.", x.size(0))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = x.size(0) // 2
        adv1 = run_autoattack_with_fallback(adversary, x[:mid], y[:mid], logger)
        adv2 = run_autoattack_with_fallback(adversary, x[mid:], y[mid:], logger)
        return torch.cat([adv1, adv2], dim=0)


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_id",
        "timestamp",
        "model_name",
        "checkpoint_path",
        "state_dict_used",
        "epoch",
        "attack_norm",
        "epsilon_input",
        "epsilon_eval",
        "clean_acc",
        "robust_acc",
        "num_images",
        "batch_size",
        "num_batches",
        "seed",
        "selection_json",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_array_index(args: argparse.Namespace) -> int:
    if args.array_index is not None:
        return args.array_index
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise SystemExit("Missing --array-index and SLURM_ARRAY_TASK_ID")
    return int(raw)


def run_autoattack_sweep_for_checkpoint(
    checkpoint_path: str | Path,
    model_dir: str | Path,
    val_dir: str | Path,
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    num_batches: int = 2,
    num_workers: int = 8,
    seed: int = 0,
    output_csv: str = OUTPUT_CSV,
    force: bool = False,
    dry_run: bool = False,
    max_settings: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    epoch: Optional[int] = None,
) -> Optional[Path]:
    checkpoint_path = Path(checkpoint_path)
    model_dir = Path(model_dir)
    val_dir = Path(val_dir)
    logger = logger or setup_logger(model_dir, dry_run)
    csv_path = model_dir / output_csv

    if is_complete_output(csv_path, max_settings, checkpoint_path=checkpoint_path, model_name=model_dir.name) and not force:
        logger.info("Skipping complete matching sweep CSV: %s", csv_path)
        return None

    if str(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    set_seed(seed)
    device_obj = torch.device(device)
    model, eval_cfg, state_key = load_model(checkpoint_path, device_obj)
    loader, selected_indices, selected_targets = build_selected_loader(
        val_dir,
        eval_cfg,
        batch_size,
        num_batches,
        num_workers,
        seed,
    )
    validate_raw_batch(loader)

    logger.info(
        "selected %d images from %d unique classes; first_indices=%s",
        len(selected_indices),
        len(set(selected_targets)),
        selected_indices[:10],
    )

    epoch_value = epoch if epoch is not None else extract_epoch_from_checkpoint(checkpoint_path)
    if dry_run:
        print(json.dumps({
            "run_id": RUN_ID,
            "model": model_dir.name,
            "checkpoint": str(checkpoint_path),
            "epoch": epoch_value,
            "num_images": len(selected_indices),
            "unique_classes": len(set(selected_targets)),
            "selected_indices_head": selected_indices[:20],
            "settings": expected_settings(max_settings),
        }, indent=2))
        return None

    selection_args = SimpleNamespace(seed=seed, batch_size=batch_size, num_batches=num_batches)
    save_selection(model_dir, selected_indices, selected_targets, selection_args)
    clean_acc = clean_accuracy(model, loader, device_obj)
    logger.info("clean_acc=%.4f", clean_acc * 100.0)

    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    out_rows = []
    for norm, eps_input, epsilon in expected_settings(max_settings):
        set_seed(seed)
        logger.info("running AutoAttack norm=%s epsilon_input=%s epsilon_eval=%s", norm, eps_input, epsilon)
        robust_acc = run_autoattack_setting(model, loader, device_obj, norm, epsilon, seed, logger)
        out_rows.append({
            "run_id": RUN_ID,
            "timestamp": timestamp,
            "model_name": model_dir.name,
            "checkpoint_path": str(checkpoint_path),
            "state_dict_used": state_key,
            "epoch": epoch_value,
            "attack_norm": norm,
            "epsilon_input": eps_input,
            "epsilon_eval": epsilon,
            "clean_acc": clean_acc * 100.0,
            "robust_acc": robust_acc * 100.0,
            "num_images": len(selected_indices),
            "batch_size": batch_size,
            "num_batches": num_batches,
            "seed": seed,
            "selection_json": str(model_dir / SELECTION_JSON),
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_rows(csv_path, out_rows)
    logger.info("saved %d rows to %s", len(out_rows), csv_path)
    return csv_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows = eligible_models(args.models_root)

    if args.list_models:
        for idx, row in enumerate(rows):
            print(f"{idx}\t{row['model_dir'].name}\tepoch={row['epoch']}\tckpt={row['checkpoint']}")
        print(f"eligible_models={len(rows)}")
        return

    if args.model_name is not None:
        matches = [row for row in rows if row["model_dir"].name == args.model_name]
        if not matches:
            raise SystemExit(f"Requested model is not eligible or missing checkpoint: {args.model_name}")
        idx = next(i for i, row in enumerate(rows) if row["model_dir"].name == args.model_name)
        selected = matches[0]
    else:
        idx = selected_array_index(args)
        if idx < 0 or idx >= len(rows):
            print(f"[INFO] array index {idx} out of range for {len(rows)} eligible models")
            return
        selected = rows[idx]
    model_dir = selected["model_dir"]
    logger = setup_logger(model_dir, args.dry_run)

    logger.info("eligible_models=%d selected_index=%d", len(rows), idx)
    logger.info("model=%s epoch=%s checkpoint=%s", model_dir.name, selected["epoch"], selected["checkpoint"])

    run_autoattack_sweep_for_checkpoint(
        checkpoint_path=selected["checkpoint"],
        model_dir=model_dir,
        val_dir=args.val_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_workers=args.num_workers,
        seed=args.seed,
        output_csv=args.output_csv,
        force=args.force,
        dry_run=args.dry_run,
        max_settings=args.max_settings,
        logger=logger,
        epoch=selected["epoch"],
    )


if __name__ == "__main__":
    main()
