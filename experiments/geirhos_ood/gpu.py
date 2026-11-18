"""Sharing the GPU with another job: refuse when memory is short, cap ourselves, back off on OOM.

The 4090 can be shared with a concurrent job (e.g. epsilon_bounded_contstim's
``eval_full_val.py``). This process must never take memory that job may need:

- ``check_free_memory``: before loading a model, require free >= need + margin.
- ``apply_budget``: cap this process's caching allocator at a fixed number of GB with
  ``torch.cuda.set_per_process_memory_fraction``. Going over raises OOM in *this* process;
  it cannot push the other job out.
- ``forward_with_backoff``: on an OOM, free the cache, split the batch in half and retry,
  down to ``min_batch``. Results are identical to one big batch (inference, eval mode).
"""

from __future__ import annotations

from typing import Callable

import torch

GB = 1024 ** 3


class GPUBusyError(RuntimeError):
    pass


def check_free_memory(need_gb: float, margin_gb: float,
                      mem_get_info: Callable[[], tuple[int, int]] | None = None) -> tuple[float, float]:
    """Return (free_gb, total_gb); raise GPUBusyError if free < need + margin."""
    free, total = (mem_get_info or torch.cuda.mem_get_info)()
    free_gb, total_gb = free / GB, total / GB
    if free_gb < need_gb + margin_gb:
        raise GPUBusyError(f"only {free_gb:.1f} GB free of {total_gb:.1f} GB; need {need_gb:.1f} + "
                           f"{margin_gb:.1f} GB margin")
    return free_gb, total_gb


def budget_fraction(budget_gb: float, total_bytes: int) -> float:
    if budget_gb <= 0:
        raise ValueError("budget must be positive")
    return min(1.0, budget_gb * GB / total_bytes)


def apply_budget(budget_gb: float, device: torch.device,
                 total_bytes: int | None = None,
                 setter: Callable[[float, torch.device], None] | None = None) -> float:
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())   # the setter needs an explicit index
    total = total_bytes if total_bytes is not None else torch.cuda.get_device_properties(device).total_memory
    fraction = budget_fraction(budget_gb, total)
    (setter or torch.cuda.set_per_process_memory_fraction)(fraction, device)
    return fraction


def _is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def forward_with_backoff(fn: Callable[[torch.Tensor], torch.Tensor], batch: torch.Tensor, min_batch: int = 4,
                         on_backoff: Callable[[int], None] | None = None) -> torch.Tensor:
    """fn(batch) with OOM -> halve-and-retry; re-raises once a chunk of min_batch still fails."""
    try:
        return fn(batch)
    except RuntimeError as err:
        if not _is_oom(err) or batch.shape[0] <= min_batch:
            raise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    half = batch.shape[0] // 2
    if on_backoff:
        on_backoff(half)
    return torch.cat([forward_with_backoff(fn, batch[:half], min_batch, on_backoff),
                      forward_with_backoff(fn, batch[half:], min_batch, on_backoff)])
