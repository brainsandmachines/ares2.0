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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ares.utils.validate import validate


DEFAULT_PGD_EPS = "0,0.01,0.03,0.05,0.1,0.25,0.5,1,3,5"
DEFAULT_PGD_NORMS = "linf,l2,l1"
DEFAULT_PGD_ATTACK_STEPS = 10
DEFAULT_PGD_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_DETECT_BATCHES = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)

LINF_DIVISOR = 255.0
L1_MULTIPLIER = 255.0 / 2.0


class LogitsOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PGD eval for Madry original ResNet50 checkpoints")
    parser.add_argument("--checkpoint", required=True, help="Path to one .ckpt file")
    parser.add_argument("--val-dir", required=True, help="ImageNet val root (ImageFolder)")
    parser.add_argument("--out-dir", required=True, help="Output directory for logs/csv")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--pgd-eps", default=DEFAULT_PGD_EPS)
    parser.add_argument("--pgd-norms", default=DEFAULT_PGD_NORMS)
    parser.add_argument("--pgd-attack-steps", type=int, default=DEFAULT_PGD_ATTACK_STEPS)
    parser.add_argument("--pgd-attack-restarts", type=int, default=3)
    parser.add_argument("--pgd-batch-size", type=int, default=DEFAULT_PGD_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--pgd-max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-mode", choices=["auto", "normalized", "raw"], default="normalized")
    parser.add_argument("--detect-batches", type=int, default=DEFAULT_DETECT_BATCHES)
    parser.add_argument(
        "--skip-auto-probe-if-normalizer",
        dest="skip_auto_probe_if_normalizer",
        action="store_true",
        help="When --input-mode=auto and checkpoint has normalizer keys, force raw mode and skip probe.",
    )
    parser.add_argument(
        "--no-skip-auto-probe-if-normalizer",
        dest="skip_auto_probe_if_normalizer",
        action="store_false",
        help="Do not bypass probe even when checkpoint has normalizer keys.",
    )
    parser.set_defaults(skip_auto_probe_if_normalizer=True)
    return parser.parse_args()


def setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"madry_resnet50_pgd_eval-{dt.datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger("madry_resnet50_pgd_eval")
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_list(value: str) -> List[str]:
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def parse_model_meta(ckpt_path: Path) -> Dict[str, Optional[float]]:
    name = ckpt_path.name.lower()
    m = re.search(r"resnet50_([a-z0-9]+)_eps([0-9]*\.?[0-9]+)\.ckpt$", name)
    train_norm = m.group(1) if m else "unknown"
    train_eps = float(m.group(2)) if m else None
    return {
        "model_name": ckpt_path.stem,
        "train_norm": train_norm,
        "train_eps": train_eps,
    }


def _looks_like_state_dict(d: Dict) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    tensor_items = sum(1 for k, v in d.items() if isinstance(k, str) and torch.is_tensor(v))
    return tensor_items > 0


def _extract_state_dict_payload(ckpt_obj) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                if _looks_like_state_dict(ckpt_obj[k]):
                    return ckpt_obj[k]
        if _looks_like_state_dict(ckpt_obj):
            return ckpt_obj
    raise ValueError("Could not find a valid state-dict-like payload in checkpoint")


def _remove_known_prefix(k: str) -> str:
    for prefix in (
        "module.model.",
        "model.",
        "module.attacker.model.",
        "attacker.model.",
        "module.attacker.",
        "attacker.",
        "module.",
    ):
        if k.startswith(prefix):
            return k[len(prefix):]
    return k


def sanitize_state_dict(raw_state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool, str]:
    cleaned_original: Dict[str, torch.Tensor] = {}
    cleaned_attacker: Dict[str, torch.Tensor] = {}
    normalizer_in_ckpt = False

    for k, v in raw_state_dict.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            continue

        low = k.lower()
        if (
            low.startswith("module.normalizer.")
            or low.startswith("normalizer.")
            or low.startswith("module.attacker.normalize.")
            or low.startswith("attacker.normalize.")
        ):
            normalizer_in_ckpt = True
            continue

        ck = _remove_known_prefix(k)
        if ck.startswith("normalizer.") or ck.startswith("normalize."):
            normalizer_in_ckpt = True
            continue

        low = k.lower()
        if ".attacker.model." in low or low.startswith("attacker.model."):
            cleaned_attacker[ck] = v
        else:
            cleaned_original[ck] = v

    if cleaned_original:
        return cleaned_original, normalizer_in_ckpt, "original_model"
    return cleaned_attacker, normalizer_in_ckpt, "attacker_model"


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, bool, str]:
    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_payload = _extract_state_dict_payload(ckpt_obj)
    state_dict, normalizer_in_ckpt, state_source = sanitize_state_dict(state_payload)

    model = models.resnet50(weights=None, num_classes=1000)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Strict state_dict load failed for {ckpt_path}: {exc}") from exc

    model = LogitsOnlyWrapper(model).to(device).eval()
    return model, normalizer_in_ckpt, state_source


def build_loader(val_dir: str, batch_size: int, num_workers: int, normalized: bool) -> DataLoader:
    tfms = [Resize(256), CenterCrop(224), ToTensor()]
    if normalized:
        tfms.append(Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    ds = datasets.ImageFolder(root=val_dir, transform=Compose(tfms))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def maybe_limited_loader(loader: DataLoader, max_batches: Optional[int]):
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


def evaluate_clean(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: Optional[int]) -> Dict[str, float]:
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


def detect_input_mode(
    model: torch.nn.Module,
    val_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    probe_batches: int,
    normalizer_in_ckpt: bool,
    logger: logging.Logger,
) -> Tuple[str, Dict[str, float]]:
    probe_scores: Dict[str, float] = {}

    for mode in ("normalized", "raw"):
        loader = build_loader(val_dir, batch_size, num_workers, normalized=(mode == "normalized"))
        metrics = evaluate_clean(model, loader, device, max_batches=probe_batches)
        probe_scores[mode] = metrics["clean_top1"]
        logger.info("Probe mode=%s clean_top1=%.4f clean_top5=%.4f", mode, metrics["clean_top1"], metrics["clean_top5"])

    margin = probe_scores["normalized"] - probe_scores["raw"]
    if abs(margin) < 0.2:
        selected = "normalized" if normalizer_in_ckpt else "raw"
        logger.info(
            "Probe tie (|margin|<0.2). Selected mode=%s using normalizer_in_ckpt=%s",
            selected,
            normalizer_in_ckpt,
        )
    else:
        selected = "normalized" if margin > 0 else "raw"
        logger.info("Probe selected mode=%s by higher clean_top1 margin=%.4f", selected, margin)

    return selected, probe_scores


def make_validate_args(
    norm: str,
    eps_eval: float,
    attack_steps: int,
    attack_restarts: int,
    input_mode: str,
) -> SimpleNamespace:
    mean = IMAGENET_MEAN if input_mode == "normalized" else IDENTITY_MEAN
    std = IMAGENET_STD if input_mode == "normalized" else IDENTITY_STD
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
        attack_restarts=attack_restarts,
        attack_use_best=True,
        attack_random_start=True,
        disable_attack_step_warmup=True,
        attack_norm=norm,
        attack_criterion="regular",
        amp_version="",
        std=std,
        mean=mean,
        log_interval=50,
    )


def save_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_pgd_sweep(
    model: torch.nn.Module,
    ckpt_path: Path,
    val_dir: str,
    device: torch.device,
    input_mode: str,
    normalizer_in_ckpt: bool,
    batch_size: int,
    num_workers: int,
    eps_values: List[float],
    norms: List[str],
    attack_steps: int,
    attack_restarts: int,
    max_batches: Optional[int],
    logger: logging.Logger,
) -> List[Dict[str, float]]:
    model.eval()
    normalized = input_mode == "normalized"
    loader = build_loader(val_dir, batch_size, num_workers, normalized=normalized)
    eval_loader = maybe_limited_loader(loader, max_batches)

    clean_metrics = evaluate_clean(model, eval_loader, device, max_batches)
    logger.info("Clean metrics mode=%s Acc@1=%.4f Acc@5=%.4f", input_mode, clean_metrics["clean_top1"], clean_metrics["clean_top5"])

    meta = parse_model_meta(ckpt_path)
    rows: List[Dict[str, float]] = []

    for norm in norms:
        for eps_input in eps_values:
            if norm == "linf":
                eps_eval = eps_input / LINF_DIVISOR
            elif norm == "l1":
                eps_eval = eps_input * L1_MULTIPLIER
            else:
                eps_eval = eps_input

            v_args = make_validate_args(
                norm=norm,
                eps_eval=eps_eval,
                attack_steps=attack_steps,
                attack_restarts=attack_restarts,
                input_mode=input_mode,
            )
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
                "model_name": meta["model_name"],
                "checkpoint_path": str(ckpt_path),
                "category": "madry_orig",
                "train_norm": meta["train_norm"],
                "train_eps": meta["train_eps"],
                "attack_norm": norm,
                "epsilon_input": float(eps_input),
                "epsilon_eval": float(eps_eval),
                "attack_steps": int(attack_steps),
                "attack_restarts": int(attack_restarts),
                "attack_step": float(v_args.attack_step),
                "clean_top1": float(clean_metrics["clean_top1"]),
                "clean_top5": float(clean_metrics["clean_top5"]),
                "adv_top1": float(metrics["advtop1"]),
                "adv_top5": float(metrics["advtop5"]),
                "clean_loss": float(metrics["loss"]),
                "adv_loss": float(metrics["advloss"]),
                "input_mode_detected": input_mode,
                "normalizer_in_ckpt": bool(normalizer_in_ckpt),
            }
            rows.append(row)
            logger.info(
                "PGD done model=%s norm=%s eps_input=%s eps_eval=%.8f adv_top1=%.4f",
                meta["model_name"],
                norm,
                eps_input,
                eps_eval,
                row["adv_top1"],
            )

    return rows


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if not os.path.isdir(args.val_dir):
        raise FileNotFoundError(f"Validation dir not found: {args.val_dir}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger = setup_logger(args.out_dir)
    set_seed(args.seed)

    eps_values = parse_float_list(args.pgd_eps)
    norms = parse_csv_list(args.pgd_norms)

    logger.info("Loading checkpoint: %s", ckpt_path)
    model, normalizer_in_ckpt, state_source = load_model_from_ckpt(ckpt_path, device=torch.device(args.device))
    logger.info("Loaded state source: %s", state_source)
    logger.info("Detected normalizer keys in ckpt: %s", normalizer_in_ckpt)

    if args.input_mode == "auto":
        if normalizer_in_ckpt and args.skip_auto_probe_if_normalizer:
            selected_mode = "raw"
            logger.info(
                "Auto mode short-circuit: checkpoint normalizer keys detected -> forcing raw input and skipping probe."
            )
        else:
            selected_mode, probe_scores = detect_input_mode(
                model=model,
                val_dir=args.val_dir,
                device=torch.device(args.device),
                batch_size=args.pgd_batch_size,
                num_workers=args.num_workers,
                probe_batches=args.detect_batches,
                normalizer_in_ckpt=normalizer_in_ckpt,
                logger=logger,
            )
            logger.info("Auto input-mode selected: %s (probe=%s)", selected_mode, probe_scores)
    else:
        selected_mode = args.input_mode
        logger.info("Input-mode forced by user: %s", selected_mode)

    rows = evaluate_pgd_sweep(
        model=model,
        ckpt_path=ckpt_path,
        val_dir=args.val_dir,
        device=torch.device(args.device),
        input_mode=selected_mode,
        normalizer_in_ckpt=normalizer_in_ckpt,
        batch_size=args.pgd_batch_size,
        num_workers=args.num_workers,
        eps_values=eps_values,
        norms=norms,
        attack_steps=args.pgd_attack_steps,
        attack_restarts=args.pgd_attack_restarts,
        max_batches=args.pgd_max_batches,
        logger=logger,
    )

    out_csv = os.path.join(args.out_dir, "pgd_validation_results.csv")
    save_csv(rows, out_csv)
    logger.info("Saved rows=%d to %s", len(rows), out_csv)


if __name__ == "__main__":
    main()
