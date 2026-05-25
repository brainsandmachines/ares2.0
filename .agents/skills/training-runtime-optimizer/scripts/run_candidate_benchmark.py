#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
from omegaconf import OmegaConf
from timm.models import model_parameters

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import robust_training.adversarial_training as advt
import ares.utils.adv as adv_mod
import ares.utils.dataset as dataset_mod
from ares.utils.adv import adv_generator, trades_adv_generator, v1_adv_generator, v1_trades_adv_generator
from ares.utils.dataset import build_dataset as real_build_dataset
from ares.utils.gradnorm import compute_gradnorm_alpha

@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    hydra_overrides: Dict[str, object]


def _pixel_protocol(norm: str, criterion: str, trades_random_start_override: bool | None = None) -> ProtocolSpec:
    attack_eps = {
        "linf": 1.0,
        "l2": 2.0,
        "l1": 2.0,
    }[norm]
    overrides = {
        "training": {"epochs": 1, "batch_size": 32, "model_ema": False},
        "attacks": {
            "advtrain": True,
            "attack_domain": "pixel",
            "attack_norm": norm,
            "attack_criterion": criterion,
            "attack_eps": attack_eps,
            "attack_step": None,
            "attack_it": 3,
            "gradnorm": False,
        },
        "model": {"model": "convnext_small", "resume": "", "experiment_num": 1},
    }
    if trades_random_start_override is not None:
        overrides["attacks"]["trades_random_start_override"] = trades_random_start_override
    suffix = ""
    if criterion == "trades" and trades_random_start_override is not None:
        suffix = "_rs_on" if trades_random_start_override else "_rs_off"
    return ProtocolSpec(
        name=f"pixel_{norm}_{criterion}{suffix}",
        hydra_overrides=overrides,
    )


PROTOCOLS: Dict[str, ProtocolSpec] = {
    "pixel_linf_madry": _pixel_protocol("linf", "madry"),
    "pixel_l2_madry": _pixel_protocol("l2", "madry"),
    "pixel_l1_madry": _pixel_protocol("l1", "madry"),
    "pixel_linf_trades_rs_on": _pixel_protocol("linf", "trades", True),
    "pixel_linf_trades_rs_off": _pixel_protocol("linf", "trades", False),
    "pixel_l2_trades_rs_on": _pixel_protocol("l2", "trades", True),
    "pixel_l2_trades_rs_off": _pixel_protocol("l2", "trades", False),
    "pixel_l1_trades_rs_on": _pixel_protocol("l1", "trades", True),
    "pixel_l1_trades_rs_off": _pixel_protocol("l1", "trades", False),
    "gradnorm_l2": ProtocolSpec(
        name="gradnorm_l2",
        hydra_overrides={
            "training": {"epochs": 1, "batch_size": 32, "model_ema": False},
            "attacks": {
                "advtrain": False,
                "gradnorm": True,
                "attack_domain": "pixel",
                "attack_norm": "linf",
                "attack_eps": 8.0,
                "attack_criterion": "madry",
                "gradnorm_penalty_norm": "l2",
            },
            "model": {"model": "convnext_small", "resume": "", "experiment_num": 1},
        },
    ),
    "v1_l2_madry": ProtocolSpec(
        name="v1_l2_madry",
        hydra_overrides={
            "training": {"epochs": 1, "batch_size": 32, "model_ema": False},
            "attacks": {
                "advtrain": True,
                "attack_domain": "v1_feature",
                "attack_norm": "l2",
                "attack_criterion": "madry",
                "v1_attack_eps": 2.0,
                "v1_attack_step": None,
                "v1_attack_it": 3,
            },
            "model": {"model": "convnext_small_v1", "resume": "", "experiment_num": 1},
        },
    ),
    "v1_l2_trades": ProtocolSpec(
        name="v1_l2_trades",
        hydra_overrides={
            "training": {"epochs": 1, "batch_size": 32, "model_ema": False},
            "attacks": {
                "advtrain": True,
                "attack_domain": "v1_feature",
                "attack_norm": "l2",
                "attack_criterion": "trades",
                "v1_attack_eps": 2.0,
                "v1_attack_step": None,
                "v1_attack_it": 3,
            },
            "model": {"model": "convnext_small_v1", "resume": "", "experiment_num": 1},
        },
    ),
}


EXAMPLE_HARNESS_BEHAVIORS = {
    "baseline",
    "dataloader_tuned",
    "channels_last",
    "torch_compile",
    "torch_compile_no_channels_last",
    "torch_compile_reduce_overhead_no_channels_last",
    "zero_grad_set_to_none",
    "adamw_fused",
    "adamw_foreach",
    "cudnn_benchmark",
    "cached_pixel_norm_tensors",
    "trades_cached_norm_tensors",
    "channels_last_reshape_attack_flatten",
    "madry_attack_autograd_grad",
    "cached_attack_criterion",
    "attack_freeze_model_params",
    "attack_pre_zero_grad",
    "adamw_fused_channels_last",
    "adamw_fused_cudnn_benchmark",
    "channels_last_cudnn_benchmark",
    "adamw_fused_channels_last_cudnn_benchmark",
    "madry_attack_autograd_grad_attack_freeze_model_params",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run additive runtime benchmark candidate for ares training")
    parser.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    parser.add_argument(
        "--candidate",
        required=True,
        help=(
            "Candidate name. Some example names have additive behavior in this script; other names are "
            "accepted so agents can add narrow open-ended harness logic without a fixed enum."
        ),
    )
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measured-iters", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def candidate_includes(candidate: str, behavior: str) -> bool:
    return candidate == behavior or behavior in candidate


def compose_args(protocol_name: str, cli_args: argparse.Namespace) -> argparse.Namespace:
    root = ROOT / "robust_training" / "configs"
    top = OmegaConf.load(root / "config.yaml")
    top_dict = OmegaConf.to_container(top, resolve=True)
    top_dict.pop("defaults", None)

    cfg = OmegaConf.create(
        {
            **top_dict,
            "training": OmegaConf.load(root / "training" / "convnext_small.yaml"),
            "model": OmegaConf.load(root / "model" / "convnext_small.yaml"),
            "dataset": OmegaConf.load(root / "dataset" / "imagenet.yaml"),
            "optimizer": OmegaConf.load(root / "optimizer" / "adamw.yaml"),
            "attacks": OmegaConf.load(root / "attacks" / "adv.yaml"),
            "dist": OmegaConf.load(root / "dist" / "default.yaml"),
            "lr_scheduler": OmegaConf.load(root / "lr_scheduler" / "cosine.yaml"),
        }
    )

    protocol = PROTOCOLS[protocol_name]
    if protocol_name.startswith("v1_"):
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"model": OmegaConf.load(root / "model" / "convnext_small_v1.yaml")}),
        )
    cfg = OmegaConf.merge(cfg, OmegaConf.create(protocol.hydra_overrides))
    args = argparse.Namespace(**advt._merge_groups_for_hydra(cfg))

    args.train_dir = cli_args.train_dir
    args.eval_dir = cli_args.eval_dir
    args.output_dir = cli_args.output_dir
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.seed = cli_args.seed
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    args.device_id = 0
    args.final_eval = False
    args.log_interval = max(cli_args.measured_iters // 4, 1)
    args.mixup_active = bool(getattr(args, "mixup_active", False))
    return args


class BoundedLoader:
    def __init__(self, loader, total_batches: int):
        self._loader = loader
        self._total_batches = total_batches
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.num_workers = getattr(loader, "num_workers", 0)
        self.pin_memory = getattr(loader, "pin_memory", False)
        self.sampler = getattr(loader, "sampler", None)
        self.collate_fn = getattr(loader, "collate_fn", None)

    def __len__(self) -> int:
        return self._total_batches

    def __iter__(self) -> Iterator:
        produced = 0
        loader_iter = iter(self._loader)
        while produced < self._total_batches:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self._loader)
                batch = next(loader_iter)
            yield batch
            produced += 1


class TimerCollector:
    def __init__(self, warmup_iters: int, measured_iters: int):
        self.warmup_iters = warmup_iters
        self.measured_iters = measured_iters
        self.total_iters = warmup_iters + measured_iters
        self.iter_times: List[float] = []
        self.data_times: List[float] = []
        self.batch_sizes: List[int] = []
        self.losses: List[float] = []

    def record(self, iter_time: float, data_time: float, batch_size: int, loss_value: float) -> None:
        self.iter_times.append(iter_time)
        self.data_times.append(data_time)
        self.batch_sizes.append(batch_size)
        self.losses.append(loss_value)

    def summarize(self) -> Dict[str, float]:
        if len(self.iter_times) < self.total_iters:
            raise RuntimeError(
                f"Expected {self.total_iters} iterations, captured {len(self.iter_times)}"
            )
        timed = self.iter_times[self.warmup_iters : self.total_iters]
        measured_batch_sizes = self.batch_sizes[self.warmup_iters : self.total_iters]
        loss_values = self.losses[self.warmup_iters : self.total_iters]
        avg_iter_s = sum(timed) / len(timed)
        avg_batch = sum(measured_batch_sizes) / len(measured_batch_sizes)
        images_per_sec = avg_batch / max(avg_iter_s, 1e-12)
        return {
            "avg_iter_seconds": avg_iter_s,
            "avg_data_seconds": sum(self.data_times[self.warmup_iters : self.total_iters]) / len(timed),
            "images_per_sec": images_per_sec,
            "final_loss": loss_values[-1],
        }


def patch_build_dataset(candidate: str, total_batches: int):
    def _wrapped(args, num_aug_splits=0):
        original_ordered_sampler = dataset_mod.OrderedDistributedSampler
        if not getattr(args, "distributed", False):
            dataset_mod.OrderedDistributedSampler = lambda dataset: torch.utils.data.SequentialSampler(dataset)
        try:
            loader_train, loader_eval = real_build_dataset(args, num_aug_splits)
        finally:
            dataset_mod.OrderedDistributedSampler = original_ordered_sampler
        if candidate == "dataloader_tuned":
            loader_train = rebuild_loader(loader_train, persistent_workers=args.num_workers > 0, prefetch_factor=4)
            loader_eval = rebuild_loader(loader_eval, persistent_workers=args.num_workers > 0, prefetch_factor=4)
        return BoundedLoader(loader_train, total_batches), BoundedLoader(loader_eval, min(2, total_batches))

    return _wrapped


def rebuild_loader(loader, persistent_workers: bool, prefetch_factor: int):
    if not hasattr(loader, "dataset"):
        return loader
    kwargs = {
        "dataset": loader.dataset,
        "batch_size": loader.batch_size,
        "shuffle": False,
        "num_workers": getattr(loader, "num_workers", 0),
        "sampler": getattr(loader, "sampler", None),
        "collate_fn": getattr(loader, "collate_fn", None),
        "pin_memory": getattr(loader, "pin_memory", False),
        "drop_last": getattr(loader, "drop_last", False),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(**kwargs)


def apply_candidate_args(args: argparse.Namespace, candidate: str) -> None:
    if candidate in {"channels_last", "channels_last_reshape_attack_flatten"} or candidate_includes(candidate, "channels_last"):
        args.channels_last = True
    elif candidate == "dataloader_tuned":
        args.pin_mem = True
        args.num_workers = max(args.num_workers, 8)
    elif candidate == "torch_compile":
        args.channels_last = True
    elif candidate == "zero_grad_set_to_none":
        args.channels_last = bool(getattr(args, "channels_last", False))


def apply_backend_candidate(candidate: str):
    previous = {
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
    }
    if candidate == "cudnn_benchmark" or candidate_includes(candidate, "cudnn_benchmark"):
        torch.backends.cudnn.benchmark = True
    return previous


def restore_backend_candidate(previous: Dict[str, bool]) -> None:
    torch.backends.cudnn.benchmark = previous["benchmark"]
    torch.backends.cudnn.deterministic = previous["deterministic"]


def patch_attack_flatten_methods(candidate: str):
    if candidate != "channels_last_reshape_attack_flatten":
        return None

    originals = {
        "L2Step.step": adv_mod.L2Step.step,
        "L2Step.random_perturb": adv_mod.L2Step.random_perturb,
        "L1Step.project": adv_mod.L1Step.project,
        "L1Step.step": adv_mod.L1Step.step,
        "L1Step.random_perturb": adv_mod.L1Step.random_perturb,
    }

    def l2_step(self, x, g):
        l = len(x.shape) - 1
        g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=1).reshape(-1, *([1] * l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def l2_random_perturb(self, x):
        diff = torch.rand_like(x) - 0.5
        diff = diff.reshape(diff.size(0), -1)
        diff = diff / (diff.norm(p=2, dim=1, keepdim=True) + 1e-10)
        diff = diff.reshape_as(x)
        return self.apply_bounds(x + diff * self.eps)

    def l1_project(self, x):
        diff = x - self.orig_input
        diff_flat = diff.reshape(diff.size(0), -1)
        current_l1 = torch.norm(diff_flat, p=1, dim=1, keepdim=True)
        if (current_l1 <= self.eps).all():
            return self.apply_bounds(x)

        mask_needs_proj = (current_l1 > self.eps).squeeze()
        if not mask_needs_proj.any():
            return self.apply_bounds(x)

        diff_to_proj = diff_flat[mask_needs_proj]
        abs_diff = torch.abs(diff_to_proj)
        sorted_vals, _ = torch.sort(abs_diff, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_vals, dim=1)
        rho = torch.arange(1, diff_flat.size(1) + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        theta_candidates = (cumsum - self.eps) / rho
        active_cond = (sorted_vals - theta_candidates) > 0
        num_active = torch.sum(active_cond, dim=1, keepdim=True)
        idx = torch.clamp(num_active - 1, min=0).long()
        theta_star = torch.gather(theta_candidates, 1, idx)
        projected_flat = torch.sign(diff_to_proj) * torch.clamp(abs_diff - theta_star, min=0)

        diff_flat_out = diff_flat.clone()
        diff_flat_out[mask_needs_proj] = projected_flat
        x_out = self.orig_input + diff_flat_out.reshape_as(diff)
        return self.apply_bounds(x_out)

    def l1_step(self, x, g):
        mode = str(getattr(self, "l1_step_mode", "l2_norm")).lower()

        if mode == "l1_apgd":
            g_flat = g.reshape(g.size(0), -1)
            d = g_flat.size(1)
            rho = float(getattr(self, "l1_apgd_rho", 0.05))
            rho = max(0.0, min(1.0, rho))
            k = max(1, int(rho * d))

            _, topk_idx = torch.topk(torch.abs(g_flat), k=k, dim=1, largest=True, sorted=False)
            u_flat = torch.zeros_like(g_flat)
            u_flat.scatter_(1, topk_idx, torch.sign(g_flat.gather(1, topk_idx)))

            u = u_flat.reshape_as(g)
            return x + self.step_size * u

        if mode != "l2_norm":
            raise ValueError(f"Unsupported l1_step_mode: {mode}")

        g_flat = g.reshape(g.size(0), -1)
        l2_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True).reshape(-1, 1, 1, 1)
        grad_normalized = g / (l2_norm + 1e-10)
        return x + grad_normalized * self.step_size

    def l1_random_perturb(self, x):
        diff = torch.rand_like(x) - 0.5
        diff = diff.reshape(diff.size(0), -1)
        norm = torch.norm(diff, p=1, dim=1, keepdim=True)
        diff = diff / (norm + 1e-10)
        diff = diff.reshape_as(x) * self.eps
        return self.apply_bounds(x + diff)

    adv_mod.L2Step.step = l2_step
    adv_mod.L2Step.random_perturb = l2_random_perturb
    adv_mod.L1Step.project = l1_project
    adv_mod.L1Step.step = l1_step
    adv_mod.L1Step.random_perturb = l1_random_perturb
    return originals


def restore_attack_flatten_methods(originals) -> None:
    if not originals:
        return
    adv_mod.L2Step.step = originals["L2Step.step"]
    adv_mod.L2Step.random_perturb = originals["L2Step.random_perturb"]
    adv_mod.L1Step.project = originals["L1Step.project"]
    adv_mod.L1Step.step = originals["L1Step.step"]
    adv_mod.L1Step.random_perturb = originals["L1Step.random_perturb"]


def cached_pixel_norm_tensors(args: argparse.Namespace, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (
        images.device.type,
        images.device.index,
        tuple(float(item) for item in args.std),
        tuple(float(item) for item in args.mean),
    )
    cache = getattr(args, "_timing_pixel_norm_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_timing_pixel_norm_cache", cache)
    if cache_key not in cache:
        std = torch.as_tensor(args.std, device=images.device)[None, :, None, None]
        mean = torch.as_tensor(args.mean, device=images.device)[None, :, None, None]
        cache[cache_key] = (std, mean)
    return cache[cache_key]


def cached_attack_criterion(args: argparse.Namespace) -> torch.nn.CrossEntropyLoss:
    criterion = getattr(args, "_timing_attack_criterion", None)
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
        setattr(args, "_timing_attack_criterion", criterion)
    return criterion


def cached_criterion_pixel_adv_generator(args, images, target, model, eps, attack_steps, attack_lr, random_start, use_best=True):
    std_tensor = torch.Tensor(args.std).cuda(non_blocking=True)[None, :, None, None]
    mean_tensor = torch.Tensor(args.mean).cuda(non_blocking=True)[None, :, None, None]
    images = images * std_tensor + mean_tensor

    prev_training = bool(model.training)
    model.eval()
    orig_input = images.detach().cuda(non_blocking=True)
    step = adv_mod._build_step(args, orig_input, eps, attack_lr)
    attack_criterion = cached_attack_criterion(args)

    amp_autocast = contextlib.suppress
    if args.amp_version == "native":
        amp_autocast = torch.cuda.amp.autocast

    best_loss = None
    best_x = None
    if random_start:
        images = step.random_perturb(images)
    else:
        images = orig_input.clone().detach()

    l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = adv_mod._configure_l1_step_search(args, attack_lr)

    for _ in range(attack_steps):
        images = images.clone().detach().requires_grad_(True)
        with amp_autocast():
            adv_losses = attack_criterion(model((images - mean_tensor) / std_tensor), target)

        torch.mean(adv_losses).backward()
        grad = images.grad.detach()
        with torch.no_grad():
            if l1_apgd_mode_active:
                cur_loss = float(adv_losses.item())
                if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                    l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                step.step_size = l1_cur_step
                l1_prev_loss = cur_loss

            best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, images, best_x) if use_best else (adv_losses, images)
            images = step.step(images, grad)
            images = step.project(images)

    with torch.no_grad():
        adv_losses = attack_criterion(model((images - mean_tensor) / std_tensor), target)
    best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, images, best_x) if use_best else (adv_losses, images)
    if prev_training:
        model.train()

    return (best_x - mean_tensor) / std_tensor


def cached_pixel_adv_generator(args, images, target, model, eps, attack_steps, attack_lr, random_start, use_best=True):
    std_tensor, mean_tensor = cached_pixel_norm_tensors(args, images)
    images = images * std_tensor + mean_tensor

    prev_training = bool(model.training)
    model.eval()
    orig_input = images.detach().cuda(non_blocking=True)
    step = adv_mod._build_step(args, orig_input, eps, attack_lr)
    attack_criterion = torch.nn.CrossEntropyLoss()

    amp_autocast = contextlib.suppress
    if args.amp_version == "native":
        amp_autocast = torch.cuda.amp.autocast

    best_loss = None
    best_x = None
    if random_start:
        images = step.random_perturb(images)
    else:
        images = orig_input.clone().detach()

    l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = adv_mod._configure_l1_step_search(args, attack_lr)

    for _ in range(attack_steps):
        images = images.clone().detach().requires_grad_(True)
        with amp_autocast():
            adv_losses = attack_criterion(model((images - mean_tensor) / std_tensor), target)

        torch.mean(adv_losses).backward()
        grad = images.grad.detach()
        with torch.no_grad():
            if l1_apgd_mode_active:
                cur_loss = float(adv_losses.item())
                if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                    l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                step.step_size = l1_cur_step
                l1_prev_loss = cur_loss

            best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, images, best_x) if use_best else (adv_losses, images)
            images = step.step(images, grad)
            images = step.project(images)

    with torch.no_grad():
        adv_losses = attack_criterion(model((images - mean_tensor) / std_tensor), target)
    best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, images, best_x) if use_best else (adv_losses, images)
    if prev_training:
        model.train()

    return (best_x - mean_tensor) / std_tensor


def cached_pixel_trades_adv_generator(args, images, model, eps, attack_steps, attack_lr, random_start=True, use_best=True):
    std, mean = cached_pixel_norm_tensors(args, images)
    images = images * std + mean

    prev_training = bool(model.training)
    model.eval()
    x_nat = images.detach()

    with torch.no_grad():
        nat_logits = model((x_nat - mean) / std)
        nat_probs = torch.softmax(nat_logits, dim=1).detach()

    step = adv_mod._build_step(args, x_nat, eps, attack_lr)

    if random_start:
        x_adv = step.random_perturb(x_nat)
    else:
        x_adv = x_nat.clone()

    best_loss = None
    best_x = None
    l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = adv_mod._configure_l1_step_search(args, attack_lr)

    for _ in range(attack_steps):
        x_adv.requires_grad_()
        logits_adv = model((x_adv - mean) / std)
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(logits_adv, dim=1),
            nat_probs,
            reduction="none",
        ).sum(dim=1)

        grad = torch.autograd.grad(kl.sum(), x_adv)[0]

        with torch.no_grad():
            if l1_apgd_mode_active:
                cur_loss = float(kl.mean().item())
                if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                    l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                step.step_size = l1_cur_step
                l1_prev_loss = cur_loss

            best_loss, best_x = adv_mod.replace_best(kl, best_loss, x_adv, best_x) if use_best else (kl, x_adv)
            x_adv = step.step(x_adv, grad)
            x_adv = step.project(x_adv)

    with torch.no_grad():
        logits_adv = model((x_adv - mean) / std)
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(logits_adv, dim=1),
            nat_probs,
            reduction="none",
        ).sum(dim=1)
        best_loss, best_x = adv_mod.replace_best(kl.detach(), best_loss, x_adv.detach(), best_x) if use_best else (kl, x_adv.detach())
    if prev_training:
        model.train()
    best_x = torch.clamp(best_x, 0.0, 1.0)
    return (best_x - mean) / std


def madry_autograd_pixel_adv_generator(args, images, target, model, eps, attack_steps, attack_lr, random_start, use_best=True):
    std_tensor = torch.Tensor(args.std).cuda(non_blocking=True)[None, :, None, None]
    mean_tensor = torch.Tensor(args.mean).cuda(non_blocking=True)[None, :, None, None]
    images = images * std_tensor + mean_tensor

    prev_training = bool(model.training)
    model.eval()
    orig_input = images.detach().cuda(non_blocking=True)
    step = adv_mod._build_step(args, orig_input, eps, attack_lr)
    attack_criterion = torch.nn.CrossEntropyLoss()

    amp_autocast = contextlib.suppress
    if args.amp_version == "native":
        amp_autocast = torch.cuda.amp.autocast

    best_loss = None
    best_x = None
    if random_start:
        images = step.random_perturb(images)
    else:
        images = orig_input.clone().detach()

    l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = adv_mod._configure_l1_step_search(args, attack_lr)

    for _ in range(attack_steps):
        images = images.clone().detach().requires_grad_(True)
        with amp_autocast():
            adv_losses = attack_criterion(model((images - mean_tensor) / std_tensor), target)

        grad = torch.autograd.grad(torch.mean(adv_losses), images)[0].detach()
        with torch.no_grad():
            if l1_apgd_mode_active:
                cur_loss = float(adv_losses.item())
                if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                    l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                step.step_size = l1_cur_step
                l1_prev_loss = cur_loss

            best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, images, best_x) if use_best else (adv_losses, images)
            images = step.step(images, grad)
            images = step.project(images)

    with torch.no_grad():
        adv_losses = attack_criterion(model((images - mean_tensor) / std_tensor), target)
    best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, images, best_x) if use_best else (adv_losses, images)
    if prev_training:
        model.train()

    return (best_x - mean_tensor) / std_tensor


def madry_autograd_v1_adv_generator(args, images, target, model, eps, attack_steps, attack_lr, random_start, use_best=True):
    prev_training = bool(model.training)
    model.eval()
    attack_model = adv_mod._unwrap_v1_attack_model(model)
    orig_features = attack_model.forward_v1_features(images, apply_noise=True).detach()
    step = adv_mod._build_step(args, orig_features, eps, attack_lr, clamp_min=None, clamp_max=None)
    attack_criterion = torch.nn.CrossEntropyLoss()

    amp_autocast = contextlib.suppress
    if args.amp_version == "native":
        amp_autocast = torch.cuda.amp.autocast

    best_loss = None
    best_x = None
    if random_start:
        adv_features = step.random_perturb(orig_features)
    else:
        adv_features = orig_features.clone().detach()

    l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = adv_mod._configure_l1_step_search(args, attack_lr)

    for _ in range(attack_steps):
        adv_features = adv_features.clone().detach().requires_grad_(True)
        with amp_autocast():
            adv_losses = attack_criterion(attack_model.forward_from_v1_features(adv_features), target)

        grad = torch.autograd.grad(torch.mean(adv_losses), adv_features)[0].detach()
        with torch.no_grad():
            if l1_apgd_mode_active:
                cur_loss = float(adv_losses.item())
                if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                    l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                step.step_size = l1_cur_step
                l1_prev_loss = cur_loss

            best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, adv_features, best_x) if use_best else (adv_losses, adv_features)
            adv_features = step.step(adv_features, grad)
            adv_features = step.project(adv_features)

    with torch.no_grad():
        adv_losses = attack_criterion(attack_model.forward_from_v1_features(adv_features), target)
    best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, adv_features, best_x) if use_best else (adv_losses, adv_features)
    if prev_training:
        model.train()
    return best_x.detach()


def cached_criterion_v1_adv_generator(args, images, target, model, eps, attack_steps, attack_lr, random_start, use_best=True):
    prev_training = bool(model.training)
    model.eval()
    attack_model = adv_mod._unwrap_v1_attack_model(model)
    orig_features = attack_model.forward_v1_features(images, apply_noise=True).detach()
    step = adv_mod._build_step(args, orig_features, eps, attack_lr, clamp_min=None, clamp_max=None)
    attack_criterion = cached_attack_criterion(args)

    amp_autocast = contextlib.suppress
    if args.amp_version == "native":
        amp_autocast = torch.cuda.amp.autocast

    best_loss = None
    best_x = None
    if random_start:
        adv_features = step.random_perturb(orig_features)
    else:
        adv_features = orig_features.clone().detach()

    l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = adv_mod._configure_l1_step_search(args, attack_lr)

    for _ in range(attack_steps):
        adv_features = adv_features.clone().detach().requires_grad_(True)
        with amp_autocast():
            adv_losses = attack_criterion(attack_model.forward_from_v1_features(adv_features), target)

        torch.mean(adv_losses).backward()
        grad = adv_features.grad.detach()
        with torch.no_grad():
            if l1_apgd_mode_active:
                cur_loss = float(adv_losses.item())
                if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                    l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                step.step_size = l1_cur_step
                l1_prev_loss = cur_loss

            best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, adv_features, best_x) if use_best else (adv_losses, adv_features)
            adv_features = step.step(adv_features, grad)
            adv_features = step.project(adv_features)

    with torch.no_grad():
        adv_losses = attack_criterion(attack_model.forward_from_v1_features(adv_features), target)
    best_loss, best_x = adv_mod.replace_best(adv_losses, best_loss, adv_features, best_x) if use_best else (adv_losses, adv_features)
    if prev_training:
        model.train()
    return best_x.detach()


def patch_build_model(candidate: str, compile_state: Dict[str, bool] | None = None):
    real_build_model = advt.build_model

    def _wrapped(*inner_args, **inner_kwargs):
        model = real_build_model(*inner_args, **inner_kwargs)
        if candidate in {"torch_compile", "torch_compile_no_channels_last"} and hasattr(torch, "compile"):
            model = torch.compile(model)
            if compile_state is not None:
                compile_state["applied"] = True
        elif candidate == "torch_compile_reduce_overhead_no_channels_last" and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
            if compile_state is not None:
                compile_state["applied"] = True
        return model

    return _wrapped


def patch_create_optimizer(candidate: str):
    real_create_optimizer = advt.create_optimizer_v2

    def _wrapped(model, **kwargs):
        if candidate == "adamw_fused" or candidate_includes(candidate, "adamw_fused"):
            kwargs = dict(kwargs)
            kwargs["fused"] = True
        elif candidate == "adamw_foreach":
            kwargs = dict(kwargs)
            kwargs["foreach"] = True
        optimizer = real_create_optimizer(model, **kwargs)
        if candidate == "zero_grad_set_to_none":
            original_zero_grad = optimizer.zero_grad

            def zero_grad_with_none(*args, **zero_kwargs):
                zero_kwargs.setdefault("set_to_none", True)
                return original_zero_grad(*args, **zero_kwargs)

            optimizer.zero_grad = zero_grad_with_none
        return optimizer

    return _wrapped


def patch_validate():
    def _wrapped(*_args, **_kwargs):
        return {"top1": 0.0, "loss": 0.0}

    return _wrapped


def patch_final_eval():
    def _wrapped(*_args, **_kwargs):
        return None

    return _wrapped


class NoOpCheckpointSaver:
    def __init__(self, *args, **kwargs):
        pass

    def save_checkpoint(self, epoch, metric=None):
        return metric, epoch

    def save_recovery(self, *args, **kwargs):
        return None


def patch_wandb():
    advt.wandb.init = lambda **_kwargs: None
    advt.wandb.log = lambda *_args, **_kwargs: None
    advt.wandb.util.generate_id = lambda: "timing-skill"


def patch_train_one_epoch(timer: TimerCollector, candidate: str):
    pixel_adv_generator = cached_pixel_adv_generator if candidate == "cached_pixel_norm_tensors" else adv_generator
    pixel_trades_adv_generator = (
        cached_pixel_trades_adv_generator
        if candidate in {"cached_pixel_norm_tensors", "trades_cached_norm_tensors"}
        else trades_adv_generator
    )
    v1_madry_adv_generator = v1_adv_generator
    if candidate == "madry_attack_autograd_grad" or candidate_includes(candidate, "madry_attack_autograd_grad"):
        pixel_adv_generator = madry_autograd_pixel_adv_generator
        v1_madry_adv_generator = madry_autograd_v1_adv_generator
    elif candidate == "cached_attack_criterion":
        pixel_adv_generator = cached_criterion_pixel_adv_generator
        v1_madry_adv_generator = cached_criterion_v1_adv_generator

    @contextlib.contextmanager
    def maybe_freeze_attack_params(model):
        if candidate != "attack_freeze_model_params" and not candidate_includes(candidate, "attack_freeze_model_params"):
            yield
            return

        params = list(model.parameters())
        previous = [param.requires_grad for param in params]
        try:
            for param in params:
                param.requires_grad_(False)
            yield
        finally:
            for param, requires_grad in zip(params, previous):
                param.requires_grad_(requires_grad)

    def _wrapped(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        reg_loss_fn=None,
        lr_scheduler=None,
        saver=None,
        amp_autocast=None,
        loss_scaler=None,
        model_ema=None,
        _logger=None,
        gradnorm_start_epoch=0,
    ):
        attack_domain = getattr(args, "attack_domain", "pixel")
        if attack_domain == "pixel":
            att_step = args.attack_step * min(epoch, 5) / 5
            att_eps = args.attack_eps
            att_it = args.attack_it
        else:
            att_step = args.v1_attack_step * min(epoch, 5) / 5
            att_eps = args.v1_attack_eps
            att_it = args.v1_attack_it

        model.train()
        end = time.perf_counter()
        for batch_idx, (inputs, target) in enumerate(loader):
            iter_start = time.perf_counter()
            data_time = iter_start - end
            inputs, target = inputs.cuda(non_blocking=True), target.cuda(non_blocking=True)
            model_inputs = inputs
            if args.channels_last:
                model_inputs = inputs.contiguous(memory_format=torch.channels_last)

            if candidate == "attack_pre_zero_grad" and args.advtrain:
                optimizer.zero_grad(set_to_none=True)

            if args.advtrain:
                if args.attack_criterion == "madry":
                    with maybe_freeze_attack_params(model):
                        if attack_domain == "pixel":
                            adv_inputs = pixel_adv_generator(args, inputs, target, model, att_eps, att_it, att_step, random_start=False)
                        else:
                            adv_inputs = v1_madry_adv_generator(args, inputs, target, model, att_eps, att_it, att_step, random_start=False)
                else:
                    trades_random_start = getattr(args, "trades_random_start_override", None)
                    if trades_random_start is None:
                        trades_random_start = args.attack_norm != "l1"
                    with maybe_freeze_attack_params(model):
                        if attack_domain == "pixel":
                            adv_inputs = pixel_trades_adv_generator(args, inputs, model, att_eps, att_it, att_step, random_start=trades_random_start)
                        else:
                            adv_inputs = v1_trades_adv_generator(args, inputs, model, att_eps, att_it, att_step, random_start=trades_random_start)
            else:
                adv_inputs = None

            if args.channels_last and attack_domain == "pixel" and adv_inputs is not None:
                adv_inputs = adv_inputs.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                if args.advtrain:
                    if args.attack_criterion == "madry":
                        if attack_domain == "pixel":
                            output = model(adv_inputs)
                        else:
                            attack_model = model.module if hasattr(model, "module") else model
                            output = attack_model.forward_from_v1_features(adv_inputs)
                        loss = loss_fn(output, target)
                    else:
                        output = model(model_inputs)
                        ce_loss = loss_fn(output, target)
                        if attack_domain == "pixel":
                            output_adv = model(adv_inputs)
                        else:
                            attack_model = model.module if hasattr(model, "module") else model
                            output_adv = attack_model.forward_from_v1_features(adv_inputs)
                        kl_loss = torch.nn.functional.kl_div(
                            torch.nn.functional.log_softmax(output_adv, dim=1),
                            torch.nn.functional.softmax(output, dim=1),
                            reduction="batchmean",
                        )
                        loss = ce_loss + args.trades_beta * kl_loss
                elif args.gradnorm:
                    inputs.requires_grad_(True)
                    gradnorm_inputs = inputs.contiguous(memory_format=torch.channels_last) if args.channels_last else inputs
                    output = model(gradnorm_inputs)
                    ce_loss = loss_fn(output, target)
                    gradient = torch.autograd.grad(ce_loss, inputs, create_graph=True, retain_graph=True)[0]
                    alpha = compute_gradnorm_alpha(
                        epoch,
                        batch_idx,
                        len(loader),
                        gradnorm_start_epoch,
                        getattr(args, "alpha_scale_epochs", 9.0),
                        getattr(args, "alpha_scale_init", 0.1),
                    )
                    raw_loss_reg = reg_loss_fn(gradient, inputs) * alpha
                    max_ratio = getattr(args, "gradnorm_max_reg_to_ce_ratio", 0.0)
                    if max_ratio is not None and max_ratio > 0:
                        reg_cap = max_ratio * ce_loss.detach()
                        scale = (reg_cap / raw_loss_reg.detach().clamp_min(1e-12)).clamp(max=1.0)
                        loss = ce_loss + raw_loss_reg * scale
                    else:
                        loss = ce_loss + raw_loss_reg
                else:
                    output = model(model_inputs)
                    loss = loss_fn(output, target)

            if not torch.isfinite(loss).all():
                raise ValueError(f"Loss is NaN/Inf at batch {batch_idx}")

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head="agc" in args.clip_mode),
                    create_graph=False,
                )
            else:
                loss.backward()
                optimizer.step()

            if model_ema is not None:
                model_ema.update(model)

            torch.cuda.synchronize()
            iter_end = time.perf_counter()
            timer.record(
                iter_time=iter_end - iter_start,
                data_time=data_time,
                batch_size=int(inputs.size(0)),
                loss_value=float(loss.detach().item()),
            )
            end = iter_end

        return {"loss": timer.losses[-1], "avg_iter_seconds": timer.summarize()["avg_iter_seconds"]}

    return _wrapped


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    cli_args = parse_args()
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the timing benchmark")

    args = compose_args(cli_args.protocol, cli_args)
    if cli_args.candidate not in EXAMPLE_HARNESS_BEHAVIORS:
        print(
            f"warning: candidate '{cli_args.candidate}' has no example harness behavior; "
            "running with baseline behavior unless this script was extended additively.",
            file=sys.stderr,
        )
    apply_candidate_args(args, cli_args.candidate)
    previous_backend = apply_backend_candidate(cli_args.candidate)
    attack_flatten_originals = patch_attack_flatten_methods(cli_args.candidate)
    compile_requested = candidate_includes(cli_args.candidate, "torch_compile")
    compile_available = hasattr(torch, "compile")
    compile_state = {"applied": False}

    total_iters = cli_args.warmup_iters + cli_args.measured_iters
    timer = TimerCollector(cli_args.warmup_iters, cli_args.measured_iters)

    original_build_dataset = advt.build_dataset
    original_build_model = advt.build_model
    original_create_optimizer = advt.create_optimizer_v2
    original_validate = advt.validate
    original_maybe_run_final_eval = advt._maybe_run_final_eval
    original_train_one_epoch = advt.train_one_epoch
    original_checkpoint_saver = advt.CheckpointSaver

    patch_wandb()
    advt.distributed_init = lambda _args: None
    advt.build_dataset = patch_build_dataset(cli_args.candidate, total_iters)
    advt.build_model = patch_build_model(cli_args.candidate, compile_state)
    advt.create_optimizer_v2 = patch_create_optimizer(cli_args.candidate)
    advt.validate = patch_validate()
    advt._maybe_run_final_eval = patch_final_eval()
    advt.train_one_epoch = patch_train_one_epoch(timer, cli_args.candidate)
    advt.CheckpointSaver = NoOpCheckpointSaver

    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    max_memory_before = torch.cuda.max_memory_allocated()
    status = "ok"
    error_message = ""
    try:
        advt.main(args)
    except Exception as exc:  # pragma: no cover - failure path is part of artifact handling
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - start
        max_memory = max(torch.cuda.max_memory_allocated(), max_memory_before)
        advt.build_dataset = original_build_dataset
        advt.build_model = original_build_model
        advt.create_optimizer_v2 = original_create_optimizer
        advt.validate = original_validate
        advt._maybe_run_final_eval = original_maybe_run_final_eval
        advt.train_one_epoch = original_train_one_epoch
        advt.CheckpointSaver = original_checkpoint_saver
        restore_attack_flatten_methods(attack_flatten_originals)
        restore_backend_candidate(previous_backend)

        rows = []
        summary = None
        if len(timer.iter_times) >= timer.total_iters:
            summary = timer.summarize()
            rows.append(
                {
                    "protocol": cli_args.protocol,
                    "candidate": cli_args.candidate,
                    "batch_size": cli_args.batch_size,
                    "status": status,
                    "avg_iter_seconds": f"{summary['avg_iter_seconds']:.6f}",
                    "avg_data_seconds": f"{summary['avg_data_seconds']:.6f}",
                    "images_per_sec": f"{summary['images_per_sec']:.4f}",
                    "final_loss": f"{summary['final_loss']:.6f}",
                    "max_memory_bytes": int(max_memory),
                    "elapsed_seconds": f"{elapsed:.6f}",
                }
            )
        save_summary_csv(output_dir / "metrics.csv", rows)
        payload = {
            "protocol": cli_args.protocol,
            "candidate": cli_args.candidate,
            "batch_size": cli_args.batch_size,
            "warmup_iters": cli_args.warmup_iters,
            "measured_iters": cli_args.measured_iters,
            "num_workers": cli_args.num_workers,
            "status": status,
            "error": error_message,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_bytes": int(max_memory),
            "elapsed_seconds": elapsed,
            "checkpoint_saving_disabled": True,
            "validation_disabled": True,
            "final_eval_disabled": True,
            "external_logging_disabled": True,
            "compile_requested": compile_requested,
            "compile_available": compile_available,
            "compile_applied": compile_state["applied"],
        }
        if summary is not None:
            payload.update(summary)
        (output_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        append_jsonl(output_dir / "events.jsonl", payload)

    return 0


if __name__ == "__main__":
    sys.exit(main())
