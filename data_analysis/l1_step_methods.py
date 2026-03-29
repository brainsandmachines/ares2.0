import math
from typing import Dict

import torch

from ares.utils.adv import L1Step


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1)


def l1_project(orig: torch.Tensor, candidate: torch.Tensor, eps: float) -> torch.Tensor:
    projector = L1Step(orig_input=orig, eps=eps, step_size=1.0)
    return projector.project(candidate)


def random_l1_start(orig: torch.Tensor, eps: float) -> torch.Tensor:
    starter = L1Step(orig_input=orig, eps=eps, step_size=1.0)
    return starter.random_perturb(orig)


def normalized_l2_direction(grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    flat = _flatten(grad)
    denom = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(eps)
    return grad / denom.view(-1, 1, 1, 1)


def normalized_l1_direction(grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    flat = _flatten(grad)
    denom = torch.norm(flat, p=1, dim=1, keepdim=True).clamp_min(eps)
    return grad / denom.view(-1, 1, 1, 1)


def raw_direction(grad: torch.Tensor) -> torch.Tensor:
    return grad


def topk_sign_direction(grad: torch.Tensor, rho: float) -> torch.Tensor:
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    g_flat = _flatten(grad)
    bsz, dim = g_flat.shape
    k = max(1, int(math.floor(rho * dim)))
    _, topk_idx = torch.topk(torch.abs(g_flat), k=k, dim=1, largest=True, sorted=False)
    out = torch.zeros_like(g_flat)
    topk_sign = torch.sign(g_flat.gather(1, topk_idx))
    out.scatter_(1, topk_idx, topk_sign)
    return out.view_as(grad)


def fw_onehot_update(delta: torch.Tensor, grad: torch.Tensor, eps: float, t: int) -> torch.Tensor:
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}")
    d_flat = _flatten(delta)
    g_flat = _flatten(grad)
    bsz, _ = g_flat.shape

    i_star = torch.argmax(torch.abs(g_flat), dim=1)
    v = torch.zeros_like(d_flat)
    row_idx = torch.arange(bsz, device=delta.device)
    v[row_idx, i_star] = eps * torch.sign(g_flat[row_idx, i_star])
    gamma = 2.0 / (t + 2.0)
    return ((1.0 - gamma) * d_flat + gamma * v).view_as(delta)


def per_sample_l1(delta: torch.Tensor) -> torch.Tensor:
    return torch.norm(_flatten(delta), p=1, dim=1)


def per_sample_l0(delta: torch.Tensor, threshold: float = 1e-12) -> torch.Tensor:
    return (_flatten(delta).abs() > threshold).sum(dim=1).float()


def select_best_run_indices(losses_by_run: torch.Tensor) -> torch.Tensor:
    # losses_by_run: [num_runs, num_samples]
    if losses_by_run.ndim != 2:
        raise ValueError("losses_by_run must be 2D [num_runs, num_samples]")
    return torch.argmax(losses_by_run, dim=0)


def gather_best_per_sample(values_by_run: torch.Tensor, best_run_idx: torch.Tensor) -> torch.Tensor:
    # values_by_run: [num_runs, num_samples]
    if values_by_run.ndim != 2:
        raise ValueError("values_by_run must be 2D [num_runs, num_samples]")
    sample_idx = torch.arange(values_by_run.shape[1], device=values_by_run.device)
    return values_by_run[best_run_idx, sample_idx]


def attack_success_from_preds(clean_pred: torch.Tensor, adv_pred: torch.Tensor) -> torch.Tensor:
    if clean_pred.shape != adv_pred.shape:
        raise ValueError("clean_pred and adv_pred must have the same shape")
    return (clean_pred != adv_pred).float()


def summarize_run_metrics(
    clean_pred: torch.Tensor,
    label: torch.Tensor,
    adv_pred: torch.Tensor,
    per_sample_loss: torch.Tensor,
    per_sample_l1_dist: torch.Tensor,
    per_sample_l0_count: torch.Tensor,
    runtime_sec: float,
    num_batches: int,
    eps_eval: float,
    uses_projection: bool,
) -> Dict[str, float]:
    n = int(clean_pred.numel())
    success = attack_success_from_preds(clean_pred, adv_pred)
    adv_acc = (adv_pred == label).float().mean().item() * 100.0
    clean_acc = (clean_pred == label).float().mean().item() * 100.0

    if uses_projection:
        violation = (per_sample_l1_dist > (eps_eval + 1e-5)).float().mean().item()
    else:
        violation = float("nan")

    sec_per_image = runtime_sec / max(n, 1)
    return {
        "num_images": n,
        "success_rate": success.mean().item() * 100.0,
        "runtime_sec": runtime_sec,
        "sec_per_image": sec_per_image,
        "sec_per_batch": runtime_sec / max(num_batches, 1),
        "efficiency": (success.mean().item() * 100.0) / max(sec_per_image, 1e-12),
        "mean_l1": per_sample_l1_dist.mean().item(),
        "max_l1": per_sample_l1_dist.max().item(),
        "mean_l0": per_sample_l0_count.mean().item(),
        "median_l0": per_sample_l0_count.median().item(),
        "constraint_violation_rate": violation * 100.0 if not math.isnan(violation) else float("nan"),
        "mean_final_loss": per_sample_loss.mean().item(),
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
    }
