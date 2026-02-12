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
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ares.utils.validate import validate
from data_analysis.plot_pgd_validation import save_combined_plot, save_norm_plot

try:
    from autoattack import AutoAttack
except Exception:  # pragma: no cover - only needed when AA installed
    AutoAttack = None


DEFAULT_OUT_DIR = "data_analysis/final_eval"
DEFAULT_PGD_EPS = "0.5,1,2,4,8,16"
DEFAULT_PGD_NORMS = "linf,l2,l1"
DEFAULT_PGD_ATTACK_STEPS = 10
DEFAULT_PGD_BATCH_SIZE = 32
DEFAULT_AA_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
LINF_DIVISOR = 255.0
L1_MULTIPLIER = 255.0 / 2.0


class NormalizeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean: Iterable[float], std: Iterable[float]):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(list(mean)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(list(std)).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final robustness evaluation (AutoAttack + PGD sweep)")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--checkpoint", help="Path to single checkpoint")
    group.add_argument("--models-dir", help="Directory to search for checkpoints")

    parser.add_argument("--val-dir", default=None, help="ImageNet val dir (ImageFolder)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory root")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)

    # AutoAttack
    parser.add_argument("--aa", action="store_true", help="Run AutoAttack eval")
    parser.add_argument("--aa-batch-size", type=int, default=DEFAULT_AA_BATCH_SIZE)
    parser.add_argument("--aa-norm", choices=["Linf", "L2"], default=None, help="Override AA norm")
    parser.add_argument("--aa-eps", type=float, default=None, help="Override AA epsilon")
    parser.add_argument("--aa-max-batches", type=int, default=None, help="Limit AA to N batches (debug)")

    # PGD sweep
    parser.add_argument("--pgd", action="store_true", help="Run PGD epsilon sweep")
    parser.add_argument("--pgd-eps", default=DEFAULT_PGD_EPS, help="Comma list, e.g. '0.5,1,2,4,8,16'")
    parser.add_argument("--pgd-norms", default=DEFAULT_PGD_NORMS, help="Comma list, e.g. 'linf,l2,l1'")
    parser.add_argument("--pgd-attack-steps", type=int, default=DEFAULT_PGD_ATTACK_STEPS)
    parser.add_argument("--pgd-batch-size", type=int, default=DEFAULT_PGD_BATCH_SIZE)
    parser.add_argument("--pgd-max-batches", type=int, default=None)

    parser.add_argument("--plots", action="store_true", help="Generate accuracy-vs-epsilon plots")
    parser.add_argument("--plot-x-col", default="epsilon_input", choices=["epsilon_input", "epsilon_eval"])

    parser.add_argument("--self-test", action="store_true", help="Run lightweight parser tests only")
    return parser.parse_args()


def setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"final_eval-{dt.datetime.now().strftime('%Y-%m-%d')}.log")
    logger = logging.getLogger("final_eval")
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

    train_norm_match = re.search(r"(^|[_\-])(linf|l2|l1)($|[_\-])", low)
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


def get_eval_resize(eval_cfg: SimpleNamespace) -> int:
    resize = int(round(eval_cfg.input_size / max(eval_cfg.crop_pct, 1e-6)))
    return max(resize, eval_cfg.input_size)


def build_raw_loader(val_dir: str, eval_cfg: SimpleNamespace, batch_size: int, num_workers: int) -> DataLoader:
    resize = get_eval_resize(eval_cfg)
    transform = Compose([
        Resize(resize),
        CenterCrop(eval_cfg.input_size),
        ToTensor(),
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


def build_norm_loader(val_dir: str, eval_cfg: SimpleNamespace, batch_size: int, num_workers: int) -> DataLoader:
    resize = get_eval_resize(eval_cfg)
    transform = Compose([
        Resize(resize),
        CenterCrop(eval_cfg.input_size),
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


def parse_attack_from_path(checkpoint_path: str) -> Tuple[bool, Optional[str], Optional[float]]:
    parent_name = os.path.basename(os.path.dirname(checkpoint_path)).lower()
    file_name = os.path.basename(checkpoint_path).lower()

    if "baseline" in parent_name or "baseline" in file_name:
        return True, None, None

    norm = None
    if "linf" in parent_name or "linf" in file_name:
        norm = "Linf"
    elif "l2" in parent_name or "l2" in file_name:
        norm = "L2"

    if norm is None:
        raise ValueError("Cannot detect norm from checkpoint name. Pass --aa-norm and --aa-eps.")

    m = re.search(rf"{norm.lower()}[_\-]?([0-9]*\.?[0-9]+)", parent_name)
    if not m:
        m = re.search(rf"{norm.lower()}[_\-]?([0-9]*\.?[0-9]+)", file_name)
    if not m:
        raise ValueError("Cannot extract eps from checkpoint name. Pass --aa-norm and --aa-eps.")

    eps_raw = float(m.group(1))
    eps = eps_raw / 255.0 if norm == "Linf" else eps_raw
    return False, norm, eps


def load_model_from_ckpt(ckpt_path: Path, device: torch.device, use_ema: bool = True) -> Tuple[torch.nn.Module, SimpleNamespace, str]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    eval_cfg = infer_eval_args(ckpt)
    arch = ckpt.get("arch")
    if arch is None:
        raise ValueError(f"Missing 'arch' in checkpoint {ckpt_path}")

    model = create_model(arch, pretrained=False, num_classes=eval_cfg.num_classes)
    state_key = "state_dict_ema" if use_ema and "state_dict_ema" in ckpt else "state_dict"
    if state_key not in ckpt:
        raise ValueError(f"Missing {state_key} in checkpoint {ckpt_path}")

    state_dict = strip_module_prefix(ckpt[state_key])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model, eval_cfg, state_key


def evaluate_autoattack(
    ckpt_path: Path,
    val_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    aa_norm: Optional[str],
    aa_eps: Optional[float],
    max_batches: Optional[int],
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    if AutoAttack is None:
        raise RuntimeError("AutoAttack is not installed or failed to import.")

    if aa_norm is not None and aa_eps is None:
        raise ValueError("--aa-norm requires --aa-eps")
    if aa_eps is not None and aa_norm is None:
        raise ValueError("--aa-eps requires --aa-norm")

    skip_auto, norm, eps = parse_attack_from_path(str(ckpt_path))
    if aa_norm is not None:
        skip_auto = False
        norm = aa_norm
        eps = aa_eps

    if skip_auto:
        logger.info("Skipping AutoAttack for baseline: %s", ckpt_path)
        return None

    model, eval_cfg, state_key = load_model_from_ckpt(ckpt_path, device)
    model = NormalizeWrapper(model, eval_cfg.mean, eval_cfg.std).to(device).eval()

    loader = build_raw_loader(val_dir, eval_cfg, batch_size, num_workers)
    loader = maybe_limited_loader(loader, max_batches)

    logger.info("AA clean eval: %s (state=%s)", ckpt_path, state_key)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    clean_acc = correct / max(total, 1)
    logger.info("AA clean acc: %.4f", clean_acc * 100.0)

    adversary = AutoAttack(model, norm=norm, eps=eps, version="standard", verbose=True)
    logger.info("AA robust eval: norm=%s eps=%s batches=%s", norm, eps, "full" if max_batches is None else max_batches)

    robust_correct, robust_total = 0, 0
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = _run_autoattack_with_fallback(adversary, x, y, logger)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        robust_correct += (pred == y).sum().item()
        robust_total += y.size(0)

    robust_acc = robust_correct / max(robust_total, 1)
    logger.info("AA robust acc: %.4f", robust_acc * 100.0)

    return {
        "model": ckpt_path.stem,
        "checkpoint_path": str(ckpt_path),
        "attack_norm": norm,
        "epsilon_eval": float(eps),
        "clean_acc": float(clean_acc * 100.0),
        "robust_acc": float(robust_acc * 100.0),
        "state_dict_used": state_key,
    }


def _run_autoattack_with_fallback(adversary, x, y, logger):
    try:
        return adversary.run_standard_evaluation(x, y, bs=x.size(0))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if x.size(0) == 1:
            raise
        logger.warning("AutoAttack OOM at batch=%d. Splitting batch and retrying.", x.size(0))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = x.size(0) // 2
        x1, y1 = x[:mid], y[:mid]
        x2, y2 = x[mid:], y[mid:]
        adv1 = _run_autoattack_with_fallback(adversary, x1, y1, logger)
        adv2 = _run_autoattack_with_fallback(adversary, x2, y2, logger)
        return torch.cat([adv1, adv2], dim=0)


def make_validate_args(eval_cfg: SimpleNamespace, norm: str, eps_eval: float, attack_steps: int) -> SimpleNamespace:
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


def evaluate_pgd_sweep(
    ckpt_path: Path,
    val_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    eps_values: List[float],
    norms: List[str],
    attack_steps: int,
    max_batches: Optional[int],
    logger: logging.Logger,
) -> List[Dict[str, float]]:
    model, eval_cfg, state_key = load_model_from_ckpt(ckpt_path, device)

    loader = build_norm_loader(val_dir, eval_cfg, batch_size, num_workers)
    eval_loader = maybe_limited_loader(loader, max_batches)

    clean_metrics = evaluate_clean(model, eval_loader, device, max_batches)
    logger.info("PGD clean Acc@1=%.3f Acc@5=%.3f", clean_metrics["clean_top1"], clean_metrics["clean_top5"])

    rows = []
    meta = parse_model_meta(ckpt_path.name)

    for norm in norms:
        for eps_input in eps_values:
            if norm == "linf":
                eps_eval = eps_input / LINF_DIVISOR
            elif norm == "l1":
                eps_eval = float(eps_input) * L1_MULTIPLIER
            else:
                eps_eval = float(eps_input)
            v_args = make_validate_args(eval_cfg, norm, eps_eval, attack_steps)

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
                "model_name": ckpt_path.stem.replace(".pth", ""),
                "checkpoint_path": str(ckpt_path),
                "category": meta["category"],
                "train_norm": meta["train_norm"],
                "init": meta["init"],
                "attack_norm": norm,
                "epsilon_input": float(eps_input),
                "epsilon_eval": float(eps_eval),
                "attack_steps": int(attack_steps),
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
                "PGD done %s | norm=%s eps_input=%s eps_eval=%.6f adv_top1=%.3f",
                ckpt_path.stem,
                norm,
                eps_input,
                eps_eval,
                row["adv_top1"],
            )

    return rows


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


def parse_csv_list(value: str) -> List[str]:
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def save_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_final_evaluation(
    checkpoint_path: Optional[str],
    models_dir: Optional[str],
    val_dir: str,
    device: str,
    out_dir: str,
    aa: bool,
    pgd: bool,
    aa_batch_size: int,
    aa_norm: Optional[str],
    aa_eps: Optional[float],
    aa_max_batches: Optional[int],
    pgd_batch_size: int,
    pgd_eps: List[float],
    pgd_norms: List[str],
    pgd_attack_steps: int,
    pgd_max_batches: Optional[int],
    plots: bool,
    plot_x_col: str,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Dict[str, str]:
    logger = setup_logger(out_dir)
    set_seed(0)

    if device == "cpu":
        raise RuntimeError("CPU evaluation is not supported by validate(); use --device cuda")

    if checkpoint_path:
        checkpoints = [Path(checkpoint_path)]
    else:
        checkpoints = list_checkpoints(models_dir)

    if not checkpoints:
        raise FileNotFoundError("No checkpoints found")

    aa_rows = []
    pgd_rows = []

    for ckpt_path in checkpoints:
        logger.info("=== Final eval: %s ===", ckpt_path)
        if aa:
            aa_row = evaluate_autoattack(
                ckpt_path=ckpt_path,
                val_dir=val_dir,
                device=torch.device(device),
                batch_size=aa_batch_size,
                num_workers=num_workers,
                aa_norm=aa_norm,
                aa_eps=aa_eps,
                max_batches=aa_max_batches,
                logger=logger,
            )
            if aa_row is not None:
                aa_rows.append(aa_row)

        if pgd:
            pgd_rows.extend(
                evaluate_pgd_sweep(
                    ckpt_path=ckpt_path,
                    val_dir=val_dir,
                    device=torch.device(device),
                    batch_size=pgd_batch_size,
                    num_workers=num_workers,
                    eps_values=pgd_eps,
                    norms=pgd_norms,
                    attack_steps=pgd_attack_steps,
                    max_batches=pgd_max_batches,
                    logger=logger,
                )
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outputs = {}
    if aa_rows:
        aa_csv = os.path.join(out_dir, "autoattack_results.csv")
        save_csv(aa_rows, aa_csv)
        outputs["aa_csv"] = aa_csv

    if pgd_rows:
        pgd_csv = os.path.join(out_dir, "pgd_validation_results.csv")
        save_csv(pgd_rows, pgd_csv)
        outputs["pgd_csv"] = pgd_csv

        if plots:
            plot_dir = os.path.join(out_dir, "plots")
            saved = []
            by_model: Dict[str, List[Dict]] = {}
            for row in pgd_rows:
                by_model.setdefault(row["model_name"], []).append(row)

            import pandas as pd
            for model_name, rows in by_model.items():
                df_model = pd.DataFrame(rows)
                category = str(df_model["category"].iloc[0])
                init = str(df_model["init"].iloc[0])

                saved.append(save_combined_plot(df_model, model_name, category, init, plot_dir, plot_x_col))
                for norm in ["linf", "l2"]:
                    p = save_norm_plot(df_model, model_name, category, init, norm, plot_dir, plot_x_col)
                    if p:
                        saved.append(p)

            outputs["plots_dir"] = plot_dir

    return outputs


def run_self_test() -> None:
    examples = [
        "convnext_small_madry_linf_8_init1.pth.tar",
        "resnet50_trades_l2_2_init2_model_best.tar",
        "convnext_small_baseline_init1_model_best.pth.tar",
    ]
    for e in examples:
        m = parse_model_meta(e)
        assert "category" in m and "init" in m and "train_norm" in m

    skip, norm, eps = parse_attack_from_path("/tmp/exp_linf_8/model_best.pth.tar")
    assert skip is False and norm == "Linf" and abs(eps - (8 / 255.0)) < 1e-6

    print("self-test: OK")


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if not args.checkpoint and not args.models_dir:
        raise ValueError("Provide --checkpoint or --models-dir")

    if not args.val_dir:
        raise ValueError("--val-dir is required")

    if not args.aa and not args.pgd:
        raise ValueError("Select at least one of --aa or --pgd")

    pgd_eps = parse_float_list(args.pgd_eps)
    pgd_norms = parse_csv_list(args.pgd_norms)

    run_final_evaluation(
        checkpoint_path=args.checkpoint,
        models_dir=args.models_dir,
        val_dir=args.val_dir,
        device=args.device,
        out_dir=args.out_dir,
        aa=args.aa,
        pgd=args.pgd,
        aa_batch_size=args.aa_batch_size,
        aa_norm=args.aa_norm,
        aa_eps=args.aa_eps,
        aa_max_batches=args.aa_max_batches,
        pgd_batch_size=args.pgd_batch_size,
        pgd_eps=pgd_eps,
        pgd_norms=pgd_norms,
        pgd_attack_steps=args.pgd_attack_steps,
        pgd_max_batches=args.pgd_max_batches,
        plots=args.plots,
        plot_x_col=args.plot_x_col,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
