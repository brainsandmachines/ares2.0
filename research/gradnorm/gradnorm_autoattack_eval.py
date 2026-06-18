#!/usr/bin/env python3
"""AutoAttack sweep for external ImageNet input-gradient-regularized CNNs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from autoattack import AutoAttack
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MANIFEST = ROOT / "model_manifest.yaml"
DEFAULT_VAL_DIR = Path("/groups/golan_neurogroup/bml_group/datasets/imagenet/val")
RUN_ID = "gradnorm_external_imagenet_aa_seed0_16x16_linf-l2_eps0.1-0.5-1-2-4-6-8"
EPS_INPUTS = (0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
NORMS = ("linf", "l2")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean: Iterable[float] = IMAGENET_MEAN, std: Iterable[float] = IMAGENET_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(list(mean)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(list(std)).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model-index", type=int, default=None, help="Defaults to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--external-dir", type=Path, default=ROOT / "external")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--selection-dir", type=Path, default=ROOT / "selection")
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-inputs", default=",".join(str(v) for v in EPS_INPUTS))
    parser.add_argument("--norms", default=",".join(NORMS))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-settings", type=int, default=None)
    return parser.parse_args()


def setup_logger(out_dir: Path, dry_run: bool) -> logging.Logger:
    logger = logging.getLogger("gradnorm_autoattack_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out_dir / f"autoattack-{dt.datetime.now().strftime('%Y-%m-%d')}.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)["models"]


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_norms(raw: str) -> tuple[str, ...]:
    norms = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unsupported = sorted(set(norms) - {"linf", "l2"})
    if unsupported:
        raise ValueError(f"Unsupported norms for this sweep: {unsupported}")
    return norms


def select_model(models: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.model_name:
        for idx, model in enumerate(models):
            if model["name"] == args.model_name:
                return idx, model
        raise ValueError(f"Unknown model name: {args.model_name}")
    raw_idx = args.model_index
    if raw_idx is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise SystemExit("Missing --model-index, --model-name, and SLURM_ARRAY_TASK_ID")
        raw_idx = int(raw)
    if raw_idx < 0 or raw_idx >= len(models):
        raise IndexError(f"model index {raw_idx} outside manifest size {len(models)}")
    return raw_idx, models[raw_idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eps_eval(norm: str, eps_input: float) -> float:
    if norm == "linf":
        return float(eps_input) / 255.0
    if norm == "l2":
        return float(eps_input)
    raise ValueError(f"Unsupported norm: {norm}")


def aa_norm(norm: str) -> str:
    return {"linf": "Linf", "l2": "L2"}[norm]


def expected_settings(norms: Iterable[str], eps_inputs: Iterable[float], max_settings: int | None) -> list[tuple[str, float, float]]:
    settings = [(norm, eps, eps_eval(norm, eps)) for norm in norms for eps in eps_inputs]
    return settings[:max_settings] if max_settings is not None else settings


def build_dataset(val_dir: Path) -> datasets.ImageFolder:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return datasets.ImageFolder(str(val_dir), transform=transform)


def select_balanced_indices(ds: datasets.ImageFolder, total_images: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for idx, (_, target) in enumerate(ds.samples):
        by_class.setdefault(int(target), []).append(idx)
    if total_images > len(ds.samples):
        raise ValueError(f"Need {total_images} images, found {len(ds.samples)} in {ds.root}")
    rng = random.Random(seed)
    per_class = {class_idx: rng.sample(indices, len(indices)) for class_idx, indices in by_class.items()}
    positions = {class_idx: 0 for class_idx in per_class}
    selected: list[int] = []
    while len(selected) < total_images:
        available = [class_idx for class_idx, pos in positions.items() if pos < len(per_class[class_idx])]
        for class_idx in rng.sample(available, len(available)):
            selected.append(per_class[class_idx][positions[class_idx]])
            positions[class_idx] += 1
            if len(selected) == total_images:
                break
    rng.shuffle(selected)
    return selected


def selection_path(selection_dir: Path, seed: int, batch_size: int, num_batches: int) -> Path:
    return selection_dir / f"imagenet_val_seed{seed}_{num_batches}x{batch_size}.json"


def get_or_create_selection(ds: datasets.ImageFolder, args: argparse.Namespace) -> tuple[list[int], list[int], Path]:
    path = selection_path(args.selection_dir, args.seed, args.batch_size, args.num_batches)
    total_images = args.batch_size * args.num_batches
    if path.exists():
        with path.open("r") as handle:
            payload = json.load(handle)
        selected = [int(v) for v in payload["selected_indices"]]
        if len(selected) != total_images:
            raise ValueError(f"Selection size mismatch in {path}: expected {total_images}, got {len(selected)}")
    else:
        selected = select_balanced_indices(ds, total_images=total_images, seed=args.seed)
        targets = [int(ds.samples[idx][1]) for idx in selected]
        args.selection_dir.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump({
                "seed": args.seed,
                "batch_size": args.batch_size,
                "num_batches": args.num_batches,
                "num_images": total_images,
                "selected_indices": selected,
                "selected_targets": targets,
            }, handle, indent=2)
    targets = [int(ds.samples[idx][1]) for idx in selected]
    return selected, targets, path


def make_loader(ds: datasets.ImageFolder, indices: list[int], args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        Subset(ds, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def import_tulip_resnet(external_dir: Path):
    path = external_dir / "tulip" / "imagenet" / "resnet.py"
    spec = importlib.util.spec_from_file_location("gradnorm_tulip_resnet", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import TULIP resnet from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_relu_with_gelu(module: torch.nn.Module) -> None:
    import torch.nn as nn

    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU())
        else:
            replace_relu_with_gelu(child)


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    first = next(iter(state_dict))
    if not first.startswith("module."):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def unwrap_state_dict(ckpt: Any, preferred_key: str) -> tuple[dict[str, Any], str]:
    if isinstance(ckpt, dict):
        for key in (preferred_key, "state_dict_ema", "state_dict", "model", "model_state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value), key
        if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
            return strip_module_prefix(ckpt), "<root>"
    raise ValueError("Could not find a state dict in checkpoint")


def build_rig_model() -> tuple[torch.nn.Module, str]:
    errors = []
    try:
        from timm.models import create_model

        for arch in ("tv_resnet50", "resnet50"):
            try:
                model = create_model(arch, pretrained=False, num_classes=1000)
                replace_relu_with_gelu(model)
                return model, f"timm:{arch}"
            except Exception as exc:
                errors.append(f"timm:{arch}: {exc}")
    except Exception as exc:
        errors.append(f"timm import: {exc}")
    try:
        from torchvision.models import resnet50

        model = resnet50(weights=None, num_classes=1000)
        replace_relu_with_gelu(model)
        return model, "torchvision:resnet50"
    except Exception as exc:
        errors.append(f"torchvision:resnet50: {exc}")
    raise RuntimeError("Could not build RIG model. " + " | ".join(errors))


def build_tulip_model(external_dir: Path) -> tuple[torch.nn.Module, str]:
    module = import_tulip_resnet(external_dir)
    return module.resnet50(), "tulip:resnet50"


def checkpoint_path(model_spec: dict[str, Any], models_dir: Path) -> Path:
    return models_dir / model_spec["name"] / model_spec["checkpoint"]["filename"]


def load_model(model_spec: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, str, str]:
    ckpt_path = checkpoint_path(model_spec, args.models_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if model_spec["loader"] == "rig_resnet50_gelu":
        model, builder = build_rig_model()
        state_dict, state_key = unwrap_state_dict(ckpt, "state_dict_ema")
    elif model_spec["loader"] == "tulip_resnet50":
        model, builder = build_tulip_model(args.external_dir)
        state_dict, state_key = unwrap_state_dict(ckpt, "state_dict")
    else:
        raise ValueError(f"Unsupported loader: {model_spec['loader']}")
    model.load_state_dict(state_dict, strict=True)
    wrapped = NormalizeWrapper(model).to(device).eval()
    return wrapped, builder, state_key


def clean_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def run_autoattack_with_fallback(adversary: AutoAttack, x: torch.Tensor, y: torch.Tensor, logger: logging.Logger) -> torch.Tensor:
    try:
        return adversary.run_standard_evaluation(x, y, bs=x.size(0))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or x.size(0) == 1:
            raise
        logger.warning("AutoAttack OOM at batch=%d; splitting batch.", x.size(0))
        torch.cuda.empty_cache()
        mid = x.size(0) // 2
        first = run_autoattack_with_fallback(adversary, x[:mid], y[:mid], logger)
        second = run_autoattack_with_fallback(adversary, x[mid:], y[mid:], logger)
        return torch.cat([first, second], dim=0)


def robust_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm: str,
    epsilon: float,
    seed: int,
    logger: logging.Logger,
) -> float:
    adversary = AutoAttack(model, norm=aa_norm(norm), eps=epsilon, seed=seed, version="standard", verbose=True, device=str(device))
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = run_autoattack_with_fallback(adversary, x, y, logger)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def existing_complete(csv_path: Path, settings: list[tuple[str, float, float]], run_id: str) -> bool:
    if not csv_path.exists():
        return False
    found = set()
    try:
        with csv_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("run_id") == run_id:
                    found.add((row.get("attack_norm"), float(row.get("epsilon_input", "nan"))))
    except Exception:
        return False
    return found == {(norm, eps) for norm, eps, _ in settings}


def write_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "timestamp",
        "model_name",
        "display_name",
        "checkpoint_path",
        "builder",
        "state_key",
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
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    models = load_manifest(args.manifest)
    model_index, model_spec = select_model(models, args)
    eps_inputs = parse_float_list(args.eps_inputs)
    norms = parse_norms(args.norms)
    settings = expected_settings(norms, eps_inputs, args.max_settings)
    model_result_dir = args.results_dir / model_spec["name"]
    logger = setup_logger(model_result_dir, args.dry_run)

    if not args.val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet val dir not found: {args.val_dir}")
    ds = build_dataset(args.val_dir)
    selected_indices, selected_targets, selection_json = get_or_create_selection(ds, args)

    payload = {
        "run_id": args.run_id,
        "model_index": model_index,
        "model_name": model_spec["name"],
        "checkpoint_path": str(checkpoint_path(model_spec, args.models_dir)),
        "val_dir": str(args.val_dir),
        "num_images": len(selected_indices),
        "unique_classes": len(set(selected_targets)),
        "selection_json": str(selection_json),
        "settings": settings,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    csv_path = model_result_dir / "autoattack_sweep_results.csv"
    if existing_complete(csv_path, settings, args.run_id) and not args.force:
        logger.info("Skipping complete result file: %s", csv_path)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(args.seed)
    device = torch.device(args.device)
    model, builder, state_key = load_model(model_spec, args, device)
    loader = make_loader(ds, selected_indices, args)
    x0, _ = next(iter(loader))
    if float(x0.min()) < -1e-6 or float(x0.max()) > 1.0 + 1e-6:
        raise ValueError("AutoAttack expects raw [0, 1] image tensors")

    logger.info("Evaluating %s on %d images", model_spec["name"], len(selected_indices))
    clean_acc = clean_accuracy(model, loader, device)
    rows = []
    for norm, eps_input, epsilon in settings:
        logger.info("AutoAttack setting norm=%s eps_input=%s eps_eval=%s", norm, eps_input, epsilon)
        robust_acc_value = robust_accuracy(model, loader, device, norm, epsilon, args.seed, logger)
        rows.append({
            "run_id": args.run_id,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "model_name": model_spec["name"],
            "display_name": model_spec["display_name"],
            "checkpoint_path": str(checkpoint_path(model_spec, args.models_dir)),
            "builder": builder,
            "state_key": state_key,
            "attack_norm": norm,
            "epsilon_input": eps_input,
            "epsilon_eval": epsilon,
            "clean_acc": clean_acc,
            "robust_acc": robust_acc_value,
            "num_images": len(selected_indices),
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "seed": args.seed,
            "selection_json": str(selection_json),
        })
        write_rows(csv_path, rows)
    if rows:
        logger.info("Wrote %s", csv_path)
    else:
        summary_path = model_result_dir / "clean_only_summary.json"
        summary = {
            "run_id": args.run_id,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "model_name": model_spec["name"],
            "display_name": model_spec["display_name"],
            "checkpoint_path": str(checkpoint_path(model_spec, args.models_dir)),
            "builder": builder,
            "state_key": state_key,
            "clean_acc": clean_acc,
            "num_images": len(selected_indices),
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "seed": args.seed,
            "selection_json": str(selection_json),
            "note": "No AutoAttack settings were run because --max-settings 0 was used.",
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Wrote clean-only summary %s", summary_path)


if __name__ == "__main__":
    main()
