import argparse
import csv
import datetime as dt
import logging
import os
import random
import re
import sys
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ares.utils.validate import validate


# ---------------------------
# Cluster-editable paths
# ---------------------------
DEFAULT_MODELS_DIR = "/mnt/data/robustness_models/for_experiment"
DEFAULT_VAL_DIR = "/mnt/data/datasets/imagenet/val"
DEFAULT_OUT_CSV = "data_analysis/pgd_validation_results.csv"
DEFAULT_LOG_PATH = "data_analysis/pgd_validation.log"


# ---------------------------
# Experiment settings
# ---------------------------
EPS_VALUES = [0.5, 1, 2, 4, 8, 16]
NORMS = ["linf", "l2", "l1"]
LINF_DIVISOR = 255  # requested: linf eps = eps / 255
L1_MULTIPLIER = 255 / 2  # requested: l1 eps = eps * (255/2)
DEFAULT_ATTACK_STEPS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PGD epsilon sweep using ares.utils.validate wrapper")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="Directory with model checkpoints")
    parser.add_argument("--val-dir", default=DEFAULT_VAL_DIR, help="ImageNet val directory (ImageFolder format)")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV path")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Log file path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for evaluation")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--attack-steps", type=int, default=DEFAULT_ATTACK_STEPS)
    parser.add_argument("--max-models", type=int, default=None, help="Optional cap for quick sanity run")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on eval batches")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", help="Use state_dict_ema when available (default)")
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false", help="Disable EMA loading and use state_dict")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", help="Run parser/path sanity tests only")
    return parser.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("pgd_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_checkpoints(models_dir: str) -> List[Path]:
    models_root = Path(models_dir)
    exts = ("*.pth", "*.pt", "*.tar", "*.pth.tar")
    files = []
    for ext in exts:
        files.extend(models_root.rglob(ext))
    unique = sorted({p.resolve() for p in files})
    return [Path(p) for p in unique]


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith("module."):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def parse_model_meta(checkpoint_name: str) -> Dict[str, str]:
    low = checkpoint_name.lower()
    category = "unknown"
    for c in ("madry", "gradnorm", "trades"):
        if c in low:
            category = c
            break

    init_match = re.search(r"init[_-]?(\d+)", low)
    init = init_match.group(1) if init_match else "unknown"

    train_norm_match = re.search(r"(^|[_\-])(linf|l2)($|[_\-])", low)
    train_norm = train_norm_match.group(2) if train_norm_match else "unknown"

    return {
        "category": category,
        "init": init,
        "train_norm": train_norm,
    }


def infer_eval_args(ckpt: Dict) -> SimpleNamespace:
    ckpt_args = ckpt.get("args", None)
    mean = tuple(getattr(ckpt_args, "mean", (0.485, 0.456, 0.406)))
    std = tuple(getattr(ckpt_args, "std", (0.229, 0.224, 0.225)))
    input_size = int(getattr(ckpt_args, "input_size", 224))
    interpolation = str(getattr(ckpt_args, "interpolation", "bicubic"))
    crop_pct = float(getattr(ckpt_args, "crop_pct", 0.875))
    num_classes = int(getattr(ckpt_args, "num_classes", 1000))

    return SimpleNamespace(
        mean=mean,
        std=std,
        input_size=input_size,
        interpolation=interpolation,
        crop_pct=crop_pct,
        num_classes=num_classes,
    )


def build_eval_loader(val_dir: str, eval_cfg: SimpleNamespace, batch_size: int, num_workers: int) -> DataLoader:
    transform = Compose([
        CenterCrop(min(eval_cfg.input_size, eval_cfg.input_size)),  # Crop to square
        Resize(eval_cfg.input_size),  # Resize to desired input size
        ToTensor(),
        Normalize(mean=eval_cfg.mean, std=eval_cfg.std),
    ])

    ds = datasets.ImageFolder(root=val_dir, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def evaluate_clean(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = None) -> Dict[str, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            top1 = logits.argmax(dim=1)
            correct1 += (top1 == target).sum().item()

            top5 = torch.topk(logits, k=5, dim=1).indices
            correct5 += (top5 == target.view(-1, 1)).any(dim=1).sum().item()
            total += target.size(0)

    if total == 0:
        return {"clean_top1": 0.0, "clean_top5": 0.0}

    return {
        "clean_top1": 100.0 * correct1 / total,
        "clean_top5": 100.0 * correct5 / total,
    }


def make_validate_args(eval_cfg: SimpleNamespace, norm: str, eps_eval: float, attack_steps: int) -> SimpleNamespace:
    # attack_step heuristic: step size per PGD iteration
    attack_step = eps_eval / max(attack_steps / 2.0, 1.0)
    return SimpleNamespace(
        channels_last=False,
        distributed=False,
        world_size=1,
        advtrain=True,
        gradnorm=False,
        attack_step=attack_step,
        attack_eps=eps_eval,
        attack_it=attack_steps,
        attack_norm=norm,
        attack_criterion="regular",
        amp_version="",
        std=eval_cfg.std,
        mean=eval_cfg.mean,
        log_interval=50,
    )


def maybe_limited_loader(loader: DataLoader, max_batches: int):
    if max_batches is None:
        return loader

    class _Limited:
        def __init__(self, base, limit):
            self.base = base
            self.limit = limit

        def __len__(self):
            return min(len(self.base), self.limit)

        def __iter__(self):
            for idx, batch in enumerate(self.base):
                if idx >= self.limit:
                    break
                yield batch

    return _Limited(loader, max_batches)


def run_self_test() -> None:
    examples = [
        "convnext_small_madry_linf_8_init1.pth.tar",
        "resnet50_trades_l2_2_init2_model_best.tar",
        "convnext_small_baseline_init1_model_best.pth.tar",
    ]
    for e in examples:
        m = parse_model_meta(e)
        assert "category" in m and "init" in m and "train_norm" in m
    print("self-test: parse_model_meta OK")


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu for dry runs.")

    logger = setup_logger(args.log_path)
    set_seed(args.seed)

    checkpoints = list_checkpoints(args.models_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {args.models_dir}")
    if args.max_models is not None:
        checkpoints = checkpoints[: args.max_models]

    logger.info("Found %d checkpoints", len(checkpoints))
    logger.info("Validation dir: %s", args.val_dir)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []
    device = torch.device(args.device)

    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.name
        model_name = ckpt_path.stem.replace(".pth", "")
        meta = parse_model_meta(ckpt_name)

        logger.info("=== Evaluating model: %s ===", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        eval_cfg = infer_eval_args(ckpt)
        arch = ckpt.get("arch")
        if arch is None:
            logger.error("Skipping %s: missing 'arch' in checkpoint", ckpt_path)
            continue

        model = create_model(arch, pretrained=False, num_classes=eval_cfg.num_classes)
        state_key = "state_dict_ema" if args.use_ema and "state_dict_ema" in ckpt else "state_dict"
        if state_key not in ckpt:
            logger.error("Skipping %s: missing %s", ckpt_path, state_key)
            continue

        state_dict = strip_module_prefix(ckpt[state_key])
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()

        loader = build_eval_loader(
            val_dir=args.val_dir,
            eval_cfg=eval_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        eval_loader = maybe_limited_loader(loader, args.max_batches)

        clean_metrics = evaluate_clean(model, eval_loader, device, max_batches=args.max_batches)
        logger.info("Clean Acc@1=%.3f Acc@5=%.3f", clean_metrics["clean_top1"], clean_metrics["clean_top5"])

        for norm in NORMS:
            for eps_input in EPS_VALUES:
                if norm == "linf":
                    eps_eval = eps_input / LINF_DIVISOR
                elif norm == "l1":
                    eps_eval = float(eps_input) * L1_MULTIPLIER
                else:
                    eps_eval = float(eps_input)
                v_args = make_validate_args(eval_cfg, norm, eps_eval, args.attack_steps)

                metrics = validate(
                    model=model,
                    loader=eval_loader,
                    loss_fn=torch.nn.CrossEntropyLoss(),
                    args=v_args,
                    amp_autocast=suppress,
                    _logger=logger,
                    epoch=1,
                )

                row = {
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "model_name": model_name,
                    "checkpoint_path": str(ckpt_path),
                    "category": meta["category"],
                    "train_norm": meta["train_norm"],
                    "init": meta["init"],
                    "attack_norm": norm,
                    "epsilon_input": float(eps_input),
                    "epsilon_eval": float(eps_eval),
                    "attack_steps": int(args.attack_steps),
                    "attack_step": float(v_args.attack_step),
                    "clean_top1": float(clean_metrics["clean_top1"]),
                    "clean_top5": float(clean_metrics["clean_top5"]),
                    "adv_top1": float(metrics["advtop1"]),
                    "adv_top5": float(metrics["advtop5"]),
                    "clean_loss": float(metrics["loss"]),
                    "adv_loss": float(metrics["advloss"]),
                    "state_dict_used": state_key,
                }
                rows.append(row)
                logger.info(
                    "Done %s | norm=%s eps_input=%s eps_eval=%.6f adv_top1=%.3f",
                    model_name,
                    norm,
                    eps_input,
                    eps_eval,
                    row["adv_top1"],
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No rows were produced.")

    fields = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    expected_rows = len(checkpoints) * len(NORMS) * len(EPS_VALUES)
    logger.info("Saved CSV: %s", args.out_csv)
    logger.info("Rows: %d (expected %d)", len(rows), expected_rows)
    if len(rows) != expected_rows:
        logger.warning("Row count mismatch. Check skipped models/errors above.")


if __name__ == "__main__":
    main()
