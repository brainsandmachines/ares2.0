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
from typing import Dict, List, Optional, Set, Tuple

import torch
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ares.utils.validate import validate


DEFAULT_MODELS_DIR = "/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models"
DEFAULT_TRAIN_DIR = "/groups/golan_neurogroup/bml_group/tomerash/datasets/imagenet/train"
DEFAULT_OUT_CSV = "data_analysis/train_accuracy_eval/madry_train_accuracy.csv"
DEFAULT_LOG_PATH = "data_analysis/train_accuracy_eval/madry_train_accuracy.log"
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8
DEFAULT_ATTACK_STEPS = 3
LINF_DIVISOR = 255.0
L1_MULTIPLIER = 255.0 / 2.0
EXCLUDED_NON_MADRY = ("gradnorm", "trades", "baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Madry model_best checkpoints on the ImageNet train set using clean and matched-budget PGD-3 accuracy"
    )
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="Root directory containing model subdirectories")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="ImageNet train directory (ImageFolder format)")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV path")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Log file path")
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="validate() requires CUDA")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-models", type=int, default=None, help="Optional cap for quick sanity runs")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on train batches")
    parser.add_argument(
        "--model-name",
        action="append",
        dest="model_names",
        default=None,
        help="Restrict evaluation to one model directory name. Repeat for multiple models.",
    )
    parser.set_defaults(use_ema=True)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", help="Use state_dict_ema when available (default)")
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false", help="Always use state_dict")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", help="Run lightweight parser tests only")
    return parser.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("madry_train_accuracy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, mode="w")
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


def get_ckpt_arg(ckpt_args, key: str, default=None):
    if ckpt_args is None:
        return default
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(key, default)
    return getattr(ckpt_args, key, default)


def infer_eval_args(ckpt: Dict) -> SimpleNamespace:
    ckpt_args = ckpt.get("args", None)
    mean = tuple(get_ckpt_arg(ckpt_args, "mean", (0.485, 0.456, 0.406)))
    std = tuple(get_ckpt_arg(ckpt_args, "std", (0.229, 0.224, 0.225)))
    input_size = int(get_ckpt_arg(ckpt_args, "input_size", 224))
    crop_pct = float(get_ckpt_arg(ckpt_args, "crop_pct", 0.875))
    num_classes = int(get_ckpt_arg(ckpt_args, "num_classes", 1000))
    return SimpleNamespace(
        mean=mean,
        std=std,
        input_size=input_size,
        crop_pct=crop_pct,
        num_classes=num_classes,
    )


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith("module."):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def get_eval_resize(eval_cfg: SimpleNamespace) -> int:
    resize = int(round(eval_cfg.input_size / max(eval_cfg.crop_pct, 1e-6)))
    return max(resize, eval_cfg.input_size)


def build_norm_loader(train_dir: str, eval_cfg: SimpleNamespace, batch_size: int, num_workers: int) -> DataLoader:
    resize = get_eval_resize(eval_cfg)
    transform = Compose([
        Resize(resize),
        CenterCrop(eval_cfg.input_size),
        ToTensor(),
        Normalize(mean=eval_cfg.mean, std=eval_cfg.std),
    ])
    ds = datasets.ImageFolder(root=train_dir, transform=transform)
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


def is_madry_dir(name: str) -> bool:
    low = (name or "").lower()
    if low == "old_models":
        return False
    return not any(token in low for token in EXCLUDED_NON_MADRY)


def parse_train_meta(model_dir_name: str) -> Dict[str, Optional[float]]:
    low = (model_dir_name or "").lower()

    norm = "unknown"
    match_norm = re.search(r"(^|[_\-])(linf|l2|l1)($|[_\-])", low)
    if match_norm:
        norm = match_norm.group(2)

    init = "unknown"
    match_init = re.search(r"init[_\-]?(\d+)", low)
    if match_init:
        init = match_init.group(1)

    eps_input = None
    match_eps = re.search(r"(?:linf|l2|l1)[_\-]?([0-9]*\.?[0-9]+)", low)
    if match_eps:
        eps_input = float(match_eps.group(1))

    return {
        "train_norm": norm,
        "init": init,
        "train_eps_input": eps_input,
    }


def constrain_eps(train_norm: str, eps_input: float) -> float:
    norm = str(train_norm).lower()
    eps = float(eps_input)
    if norm == "linf":
        return eps / LINF_DIVISOR
    if norm == "l1":
        return eps * L1_MULTIPLIER
    return eps


def recover_eps_input(train_norm: str, constrained_eps: float) -> float:
    norm = str(train_norm).lower()
    eps = float(constrained_eps)
    if norm == "linf":
        return eps * LINF_DIVISOR
    if norm == "l1":
        return eps / L1_MULTIPLIER
    return eps


def discover_madry_checkpoints(models_dir: str, allowed_names: Optional[Set[str]] = None) -> List[Path]:
    root = Path(models_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")

    allowed = set(allowed_names or [])
    checkpoints: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not is_madry_dir(child.name):
            continue
        if allowed and child.name not in allowed:
            continue
        ckpt = child / "model_best.pth.tar"
        if ckpt.is_file():
            checkpoints.append(ckpt)
            continue
        fallback = child / "model_best.pth"
        if fallback.is_file():
            checkpoints.append(fallback)

    if allowed:
        found_names = {p.parent.name for p in checkpoints}
        missing = sorted(allowed - found_names)
        if missing:
            raise FileNotFoundError(f"Requested model directories not found with model_best checkpoint: {missing}")
    return checkpoints


def load_model_from_ckpt(ckpt_path: Path, device: torch.device, use_ema: bool = True) -> Tuple[torch.nn.Module, Dict, SimpleNamespace, str]:
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
    return model, ckpt, eval_cfg, state_key


def resolve_attack_config(ckpt: Dict, model_dir_name: str) -> Tuple[str, float, float, int, Optional[float]]:
    ckpt_args = ckpt.get("args", None)
    attack_norm = get_ckpt_arg(ckpt_args, "attack_norm", None)
    attack_eps = get_ckpt_arg(ckpt_args, "attack_eps", None)
    attack_step = get_ckpt_arg(ckpt_args, "attack_step", None)

    parsed = parse_train_meta(model_dir_name)
    parsed_norm = parsed["train_norm"]
    parsed_eps_input = parsed["train_eps_input"]

    if attack_norm is None:
        if parsed_norm == "unknown":
            raise ValueError(f"Cannot infer attack norm for {model_dir_name}")
        attack_norm = parsed_norm
    attack_norm = str(attack_norm).lower()

    if attack_eps is None:
        if parsed_eps_input is None:
            raise ValueError(f"Cannot infer attack epsilon for {model_dir_name}")
        attack_eps = constrain_eps(attack_norm, parsed_eps_input)
    attack_eps = float(attack_eps)

    if attack_step is None:
        attack_step = attack_eps / max(DEFAULT_ATTACK_STEPS / 2.0, 1.0)
    attack_step = float(attack_step)

    if parsed_eps_input is None:
        parsed_eps_input = recover_eps_input(attack_norm, attack_eps)

    return attack_norm, attack_eps, attack_step, DEFAULT_ATTACK_STEPS, parsed_eps_input


def make_validate_args(eval_cfg: SimpleNamespace, attack_norm: str, attack_eps: float, attack_step: float, attack_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        channels_last=False,
        distributed=False,
        world_size=1,
        advtrain=True,
        gradnorm=False,
        attack_step=float(attack_step),
        attack_eps=float(attack_eps),
        attack_it=int(attack_steps),
        attack_restarts=1,
        attack_use_best=True,
        attack_random_start=False,
        attack_norm=str(attack_norm).lower(),
        disable_attack_step_warmup=True,
        attack_criterion="regular",
        amp_version="",
        std=eval_cfg.std,
        mean=eval_cfg.mean,
        log_interval=50,
    )


def evaluate_checkpoint(
    ckpt_path: Path,
    train_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: Optional[int],
    use_ema: bool,
    logger: logging.Logger,
) -> Dict[str, float]:
    model, ckpt, eval_cfg, state_key = load_model_from_ckpt(ckpt_path, device=device, use_ema=use_ema)
    loader = build_norm_loader(train_dir, eval_cfg, batch_size, num_workers)
    eval_loader = maybe_limited_loader(loader, max_batches)

    model_dir_name = ckpt_path.parent.name
    meta = parse_train_meta(model_dir_name)
    attack_norm, attack_eps, attack_step, attack_steps, train_eps_input = resolve_attack_config(ckpt, model_dir_name)
    validate_args = make_validate_args(eval_cfg, attack_norm, attack_eps, attack_step, attack_steps)

    metrics = validate(
        model=model,
        loader=eval_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        args=validate_args,
        amp_autocast=suppress,
        _logger=logger,
        epoch=1,
    )

    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_dir_name,
        "checkpoint_path": str(ckpt_path),
        "state_dict_used": state_key,
        "train_norm": meta["train_norm"],
        "train_eps_input": float(train_eps_input),
        "attack_norm": attack_norm,
        "attack_eps": float(attack_eps),
        "attack_step": float(attack_step),
        "attack_steps": int(attack_steps),
        "init": meta["init"],
        "clean_top1": float(metrics["top1"]),
        "clean_top5": float(metrics["top5"]),
        "adv_top1": float(metrics["advtop1"]),
        "adv_top5": float(metrics["advtop5"]),
        "clean_loss": float(metrics["loss"]),
        "adv_loss": float(metrics["advloss"]),
    }
    logger.info(
        "done %s | norm=%s eps_input=%s eps=%.6f step=%.6f clean_top1=%.3f adv_top1=%.3f",
        model_dir_name,
        attack_norm,
        f"{train_eps_input:g}",
        attack_eps,
        attack_step,
        row["clean_top1"],
        row["adv_top1"],
    )
    return row


def save_csv(rows: List[Dict], out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "timestamp",
        "model_name",
        "checkpoint_path",
        "state_dict_used",
        "train_norm",
        "train_eps_input",
        "attack_norm",
        "attack_eps",
        "attack_step",
        "attack_steps",
        "init",
        "clean_top1",
        "clean_top5",
        "adv_top1",
        "adv_top5",
        "clean_loss",
        "adv_loss",
    ]
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    meta = parse_train_meta("convnext_small_linf_8_init2")
    assert meta["train_norm"] == "linf"
    assert meta["init"] == "2"
    assert float(meta["train_eps_input"]) == 8.0
    assert abs(constrain_eps("linf", 8) - (8.0 / 255.0)) < 1e-12
    assert abs(constrain_eps("l1", 16) - (16.0 * 255.0 / 2.0)) < 1e-12
    assert is_madry_dir("convnext_small_l2_4_init1")
    assert not is_madry_dir("convnext_small_linftrades_8_init1")
    print("self-test: OK")


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.device != "cuda":
        raise RuntimeError("CPU evaluation is not supported by validate(); use --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validate(), but no CUDA device is available")
    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"Train directory does not exist: {args.train_dir}")

    logger = setup_logger(args.log_path)
    set_seed(args.seed)
    device = torch.device(args.device)

    allowed_names = set(args.model_names or [])
    checkpoints = discover_madry_checkpoints(args.models_dir, allowed_names=allowed_names)
    if args.max_models is not None:
        checkpoints = checkpoints[: args.max_models]
    if not checkpoints:
        raise FileNotFoundError(f"No Madry model_best checkpoints found under {args.models_dir}")

    logger.info("discovered %d checkpoints under %s", len(checkpoints), args.models_dir)
    if allowed_names:
        logger.info("restricted to %d requested model directories", len(allowed_names))
    logger.info(
        "train_dir=%s batch_size=%d num_workers=%d max_batches=%s use_ema=%s",
        args.train_dir,
        args.batch_size,
        args.num_workers,
        args.max_batches,
        args.use_ema,
    )

    rows: List[Dict] = []
    for ckpt_path in checkpoints:
        logger.info("evaluating %s", ckpt_path)
        row = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            train_dir=args.train_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            use_ema=args.use_ema,
            logger=logger,
        )
        rows.append(row)
        save_csv(rows, args.out_csv)

    logger.info("finished %d checkpoints", len(rows))
    logger.info("csv saved to %s", args.out_csv)


if __name__ == "__main__":
    main()
