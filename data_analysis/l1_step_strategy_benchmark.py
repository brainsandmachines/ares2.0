import argparse
import csv
import datetime as dt
import json
import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from timm.models import create_model
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.l1_step_methods import (
    fw_onehot_update,
    gather_best_per_sample,
    l1_project,
    normalized_l1_direction,
    normalized_l2_direction,
    per_sample_l0,
    per_sample_l1,
    random_l1_start,
    raw_direction,
    select_best_run_indices,
    summarize_run_metrics,
    topk_sign_direction,
)

L1_MULTIPLIER = 255.0 / 2.0
DEFAULT_MODEL_NAMES = "convnext_small_l1_2_init1,convnext_small_l2_2_init1,convnext_small_linf_2_init1"
DEFAULT_EPS_LIST = "1,2,4,8,16"
DEFAULT_RHO_LIST = "0.01,0.05,0.10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark L1 step-update strategies across pretrained models")
    parser.add_argument("--models-dir", required=True, help="Directory containing model checkpoints")
    parser.add_argument("--model-names", default=DEFAULT_MODEL_NAMES, help="Comma list of model IDs")
    parser.add_argument("--val-dir", required=True, help="ImageNet val directory (ImageFolder)")
    parser.add_argument("--out-dir", default="data_analysis/l1_step_strategy_benchmark", help="Output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--eps-list", default=DEFAULT_EPS_LIST)
    parser.add_argument("--rho-list", default=DEFAULT_RHO_LIST)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("l1_step_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(out_dir / "benchmark.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith("module."):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def infer_eval_args(ckpt: Dict) -> SimpleNamespace:
    ckpt_args = ckpt.get("args", None)
    mean = tuple(getattr(ckpt_args, "mean", (0.485, 0.456, 0.406)))
    std = tuple(getattr(ckpt_args, "std", (0.229, 0.224, 0.225)))
    input_size = int(getattr(ckpt_args, "input_size", 224))
    crop_pct = float(getattr(ckpt_args, "crop_pct", 0.875))
    num_classes = int(getattr(ckpt_args, "num_classes", 1000))
    return SimpleNamespace(mean=mean, std=std, input_size=input_size, crop_pct=crop_pct, num_classes=num_classes)


def get_eval_resize(eval_cfg: SimpleNamespace) -> int:
    resize = int(round(eval_cfg.input_size / max(eval_cfg.crop_pct, 1e-6)))
    return max(resize, eval_cfg.input_size)


def build_norm_dataset(val_dir: str, eval_cfg: SimpleNamespace) -> datasets.ImageFolder:
    resize = get_eval_resize(eval_cfg)
    transform = Compose([
        Resize(resize),
        CenterCrop(eval_cfg.input_size),
        ToTensor(),
        Normalize(mean=eval_cfg.mean, std=eval_cfg.std),
    ])
    return datasets.ImageFolder(root=val_dir, transform=transform)


def get_or_create_subset_indices(path: Path, count: int, dataset_len: int) -> List[int]:
    if path.exists():
        payload = json.loads(path.read_text())
        idx = payload.get("indices", [])
        if len(idx) >= count:
            subset = [int(i) for i in idx[:count]]
            if subset and max(subset) >= dataset_len:
                raise ValueError("Cached subset indices exceed dataset length for current run")
            return subset

    idx = list(range(min(dataset_len, count)))
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "count_requested": count,
        "count_saved": len(idx),
        "indices": idx,
    }
    path.write_text(json.dumps(payload, indent=2))
    return idx


def resolve_checkpoint(models_dir: Path, model_name: str) -> Path:
    candidates = [
        models_dir / f"{model_name}.pth.tar",
        models_dir / f"{model_name}.pth",
        models_dir / model_name / "model_best.pth.tar",
        models_dir / model_name / "model_best.pth",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    matches = sorted(models_dir.rglob(f"{model_name}*.pth*"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(f"Multiple checkpoints matched {model_name}: {[str(m) for m in matches[:5]]}")
    raise FileNotFoundError(f"No checkpoint found for model name '{model_name}' under {models_dir}")


def load_model_from_ckpt(ckpt_path: Path, device: torch.device, use_ema: bool) -> Tuple[torch.nn.Module, SimpleNamespace, str]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    eval_cfg = infer_eval_args(ckpt)
    arch = ckpt.get("arch")
    if arch is None:
        raise ValueError(f"Missing 'arch' in checkpoint {ckpt_path}")

    model = create_model(arch, pretrained=False, num_classes=eval_cfg.num_classes)
    state_key = "state_dict_ema" if use_ema and "state_dict_ema" in ckpt else "state_dict"
    if state_key not in ckpt:
        raise ValueError(f"Missing {state_key} in checkpoint {ckpt_path}")

    model.load_state_dict(strip_module_prefix(ckpt[state_key]), strict=True)
    return model.to(device).eval(), eval_cfg, state_key


def build_subset_loader(dataset: datasets.ImageFolder, subset_indices: List[int], batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, subset_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def collect_clean_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    clean_preds = []
    labels = []
    with torch.no_grad():
        for x_norm, y in loader:
            x_norm = x_norm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x_norm)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            clean_preds.append(logits.argmax(dim=1))
            labels.append(y)
    return torch.cat(clean_preds), torch.cat(labels)


def _denorm(x_norm: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    return x_norm * std_t + mean_t


def _renorm(x: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    return (x - mean_t) / std_t


def _run_attack_batch(
    model: torch.nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    method: str,
    rho: Optional[float],
    eps_eval: float,
    alpha0: float,
    attack_steps: int,
    random_start: bool,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    orig = _denorm(x_norm, mean_t, std_t).detach()
    x_adv = random_l1_start(orig, eps_eval) if random_start else orig.clone()

    best_adv = x_adv.clone()
    best_loss = None
    loss_trace_sum = [0.0 for _ in range(attack_steps)]
    alpha_cur = float(alpha0)
    prev_mean_loss = None

    for t in range(1, attack_steps + 1):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(_renorm(x_adv, mean_t, std_t))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        loss_vec = ce(logits, y)
        loss_mean = loss_vec.mean()
        loss_mean.backward()
        grad = x_adv.grad.detach()

        with torch.no_grad():
            loss_trace_sum[t - 1] = float(loss_vec.sum().item())

            if best_loss is None:
                best_loss = loss_vec.detach().clone()
                best_adv = x_adv.detach().clone()
            else:
                replace = loss_vec > best_loss
                best_loss[replace] = loss_vec[replace].detach()
                best_adv[replace] = x_adv[replace].detach()

            if method == "baseline_l2norm":
                x_adv = l1_project(orig, x_adv + alpha0 * normalized_l2_direction(grad), eps_eval)
            elif method == "l1norm_step":
                x_adv = l1_project(orig, x_adv + alpha0 * normalized_l1_direction(grad), eps_eval)
            elif method == "raw_grad_step":
                x_adv = l1_project(orig, x_adv + alpha0 * raw_direction(grad), eps_eval)
            elif method == "l1_apgd_topk":
                if rho is None:
                    raise ValueError("rho is required for l1_apgd_topk")
                direction = topk_sign_direction(grad, rho=rho)
                delta_temp = (x_adv - orig) + alpha_cur * direction
                x_adv = l1_project(orig, orig + delta_temp, eps_eval)

                current_mean_loss = float(loss_mean.item())
                if prev_mean_loss is not None and current_mean_loss <= prev_mean_loss + 1e-12:
                    alpha_cur = alpha_cur / 2.0
                prev_mean_loss = current_mean_loss
            elif method == "fw_onehot":
                delta = fw_onehot_update(x_adv - orig, grad, eps_eval, t=t)
                x_adv = torch.clamp(orig + delta, 0.0, 1.0)
            else:
                raise ValueError(f"Unsupported method {method}")

    with torch.no_grad():
        final_logits = model(_renorm(x_adv, mean_t, std_t))
        if isinstance(final_logits, (tuple, list)):
            final_logits = final_logits[0]
        final_loss = ce(final_logits, y)
        if best_loss is None:
            best_loss = final_loss.detach().clone()
            best_adv = x_adv.detach().clone()
        else:
            replace = final_loss > best_loss
            best_loss[replace] = final_loss[replace].detach()
            best_adv[replace] = x_adv[replace].detach()

        adv_logits = model(_renorm(best_adv, mean_t, std_t))
        if isinstance(adv_logits, (tuple, list)):
            adv_logits = adv_logits[0]
        adv_pred = adv_logits.argmax(dim=1)

    delta_best = best_adv - orig
    return adv_pred.detach(), best_loss.detach(), per_sample_l1(delta_best).detach(), per_sample_l0(delta_best).detach(), loss_trace_sum


def run_attack_over_subset(
    model: torch.nn.Module,
    loader: DataLoader,
    clean_preds: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    rho: Optional[float],
    eps_eval: float,
    attack_steps: int,
    random_start: bool,
    seed: int,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[Dict[str, torch.Tensor], List[float], Dict[str, float]]:
    set_seed(seed)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    alpha0 = float(eps_eval / max(attack_steps / 2.0, 1.0))

    adv_preds = []
    losses = []
    l1_dists = []
    l0_counts = []
    trace_accum = [0.0 for _ in range(attack_steps)]

    t0 = time.perf_counter()
    for x_norm, y in loader:
        x_norm = x_norm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_adv_pred, batch_loss, batch_l1, batch_l0, batch_trace = _run_attack_batch(
            model=model,
            x_norm=x_norm,
            y=y,
            method=method,
            rho=rho,
            eps_eval=eps_eval,
            alpha0=alpha0,
            attack_steps=attack_steps,
            random_start=random_start,
            mean_t=mean_t,
            std_t=std_t,
        )
        adv_preds.append(batch_adv_pred)
        losses.append(batch_loss)
        l1_dists.append(batch_l1)
        l0_counts.append(batch_l0)
        for i in range(attack_steps):
            trace_accum[i] += batch_trace[i]

    runtime_sec = time.perf_counter() - t0

    adv_pred = torch.cat(adv_preds)
    per_sample_loss = torch.cat(losses)
    per_sample_l1_dist = torch.cat(l1_dists)
    per_sample_l0_count = torch.cat(l0_counts)

    num_images = int(per_sample_loss.numel())
    mean_loss_trace = [x / max(num_images, 1) for x in trace_accum]

    metrics = summarize_run_metrics(
        clean_pred=clean_preds,
        label=labels,
        adv_pred=adv_pred,
        per_sample_loss=per_sample_loss,
        per_sample_l1_dist=per_sample_l1_dist,
        per_sample_l0_count=per_sample_l0_count,
        runtime_sec=runtime_sec,
        num_batches=len(loader),
        eps_eval=eps_eval,
        uses_projection=(method != "fw_onehot"),
    )

    return {
        "adv_pred": adv_pred,
        "loss": per_sample_loss,
        "l1": per_sample_l1_dist,
        "l0": per_sample_l0_count,
    }, mean_loss_trace, metrics


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    t = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
    idx = select_best_run_indices(t)
    assert idx.tolist() == [1, 0]
    print("self-test OK")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    set_seed(args.seed)

    eps_inputs = parse_float_list(args.eps_list)
    rho_list = parse_float_list(args.rho_list)
    model_names = parse_str_list(args.model_names)

    methods: List[Tuple[str, Optional[float]]] = [
        ("baseline_l2norm", None),
        ("l1norm_step", None),
        ("raw_grad_step", None),
    ]
    methods.extend(("l1_apgd_topk", rho) for rho in rho_list)
    methods.append(("fw_onehot", None))

    long_rows: List[Dict] = []
    agg_rows: List[Dict] = []
    trace_rows: List[Dict] = []

    shared_subset_indices: Optional[List[int]] = None

    for model_name in model_names:
        ckpt_path = resolve_checkpoint(Path(args.models_dir), model_name)
        model, eval_cfg, state_key = load_model_from_ckpt(ckpt_path, device, use_ema=args.use_ema)
        dataset = build_norm_dataset(args.val_dir, eval_cfg)

        requested_count = args.batch_size * args.num_batches
        if shared_subset_indices is None:
            subset_path = out_dir / "subset_indices.json"
            shared_subset_indices = get_or_create_subset_indices(subset_path, requested_count, len(dataset))
        elif shared_subset_indices and max(shared_subset_indices) >= len(dataset):
            raise ValueError(f"Shared subset indices exceed dataset length for model {model_name}")

        loader = build_subset_loader(dataset, shared_subset_indices or [], args.batch_size, args.num_workers)
        clean_preds, labels = collect_clean_predictions(model, loader, device)
        logger.info("Model %s clean acc on subset: %.3f", model_name, (clean_preds == labels).float().mean().item() * 100.0)

        for eps_input in eps_inputs:
            eps_eval = float(eps_input) * L1_MULTIPLIER
            for method, rho in methods:
                method_tag = method if rho is None else f"{method}_rho{rho:.2f}"
                logger.info("Running %s | model=%s eps=%.3f eps_eval=%.3f", method_tag, model_name, eps_input, eps_eval)

                run_defs = [{"run_id": "deterministic", "random_start": False, "restart_index": 0, "seed": args.seed}]
                if method != "fw_onehot":
                    run_defs.extend(
                        {
                            "run_id": f"random_restart_{i}",
                            "random_start": True,
                            "restart_index": i,
                            "seed": args.seed + i,
                        }
                        for i in range(1, 4)
                    )

                run_tensors: List[Dict[str, torch.Tensor]] = []
                runtime_values: List[float] = []
                for run in run_defs:
                    tensors, loss_trace, metrics = run_attack_over_subset(
                        model=model,
                        loader=loader,
                        clean_preds=clean_preds,
                        labels=labels,
                        method=method,
                        rho=rho,
                        eps_eval=eps_eval,
                        attack_steps=args.attack_steps,
                        random_start=run["random_start"],
                        seed=run["seed"],
                        device=device,
                        mean=eval_cfg.mean,
                        std=eval_cfg.std,
                    )
                    run_tensors.append(tensors)
                    runtime_values.append(float(metrics["runtime_sec"]))

                    ts = dt.datetime.now().isoformat(timespec="seconds")
                    long_rows.append({
                        "timestamp": ts,
                        "model_name": model_name,
                        "checkpoint_path": str(ckpt_path),
                        "state_dict_used": state_key,
                        "method": method,
                        "rho": "" if rho is None else float(rho),
                        "epsilon_input": float(eps_input),
                        "epsilon_eval": float(eps_eval),
                        "attack_steps": int(args.attack_steps),
                        "run_id": run["run_id"],
                        "random_start": bool(run["random_start"]),
                        "restart_index": int(run["restart_index"]),
                        "seed": int(run["seed"]),
                        **metrics,
                    })

                    for it, mean_loss in enumerate(loss_trace, start=1):
                        trace_rows.append({
                            "timestamp": ts,
                            "model_name": model_name,
                            "method": method,
                            "rho": "" if rho is None else float(rho),
                            "epsilon_input": float(eps_input),
                            "epsilon_eval": float(eps_eval),
                            "run_id": run["run_id"],
                            "iteration": int(it),
                            "mean_loss": float(mean_loss),
                        })

                stacked_loss = torch.stack([r["loss"] for r in run_tensors], dim=0)
                best_idx = select_best_run_indices(stacked_loss)
                best_adv_pred = gather_best_per_sample(torch.stack([r["adv_pred"].float() for r in run_tensors], dim=0), best_idx).long()
                best_loss = gather_best_per_sample(torch.stack([r["loss"] for r in run_tensors], dim=0), best_idx)
                best_l1 = gather_best_per_sample(torch.stack([r["l1"] for r in run_tensors], dim=0), best_idx)
                best_l0 = gather_best_per_sample(torch.stack([r["l0"] for r in run_tensors], dim=0), best_idx)

                agg_metrics = summarize_run_metrics(
                    clean_pred=clean_preds,
                    label=labels,
                    adv_pred=best_adv_pred,
                    per_sample_loss=best_loss,
                    per_sample_l1_dist=best_l1,
                    per_sample_l0_count=best_l0,
                    runtime_sec=float(sum(runtime_values)),
                    num_batches=len(loader),
                    eps_eval=eps_eval,
                    uses_projection=(method != "fw_onehot"),
                )
                agg_rows.append({
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "model_name": model_name,
                    "checkpoint_path": str(ckpt_path),
                    "state_dict_used": state_key,
                    "method": method,
                    "rho": "" if rho is None else float(rho),
                    "epsilon_input": float(eps_input),
                    "epsilon_eval": float(eps_eval),
                    "attack_steps": int(args.attack_steps),
                    "aggregation": "best_over_runs",
                    "num_runs": len(run_defs),
                    **agg_metrics,
                })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(out_dir / "results_long.csv", long_rows)
    write_csv(out_dir / "results_agg.csv", agg_rows)
    write_csv(out_dir / "loss_traces.csv", trace_rows)

    config = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "eps_inputs": eps_inputs,
        "rho_list": rho_list,
        "model_names": model_names,
        "methods": [{"method": m, "rho": r} for m, r in methods],
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    logger.info("Saved: %s", out_dir / "results_long.csv")
    logger.info("Saved: %s", out_dir / "results_agg.csv")
    logger.info("Saved: %s", out_dir / "loss_traces.csv")


if __name__ == "__main__":
    main()
