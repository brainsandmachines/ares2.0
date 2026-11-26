"""Per-launch seed + environment provenance for a training run directory.

Every launch of ``robust_training.adversarial_training`` (fresh start, Slurm requeue,
resume, eval-only relaunch) appends one JSON line to ``<run_dir>/seed_history.jsonl``
with the seed it used, the epoch it started from, and where it ran: git commit,
Slurm ids, GPU, library versions, and cuDNN flags. The same list is embedded in
every checkpoint's ``args.seed_history``, so it travels with the weights through
rsync and HF export.

Why a history and not one seed: when ``seed`` is null (every pre-2026-11 CSV row),
each launch draws a fresh seed, so a model trained across N requeues has N seeds,
one per epoch segment. ``runtime_config.yaml`` is overwritten each launch and keeps
only the last one. For run dirs that predate this file, the first recorded launch
backfills earlier seeds from the ``Using seed: N`` lines in ``log.txt``
(``source="backfill:log.txt"``; no start epoch or environment for those).

With an explicit seed, the train loop reseeds at every epoch start with
``derive_epoch_seed(seed, epoch)``. The RNG stream of epoch k (shuffle order, worker
augmentations, Mixup, TRADES random start) then depends only on ``(seed, epoch)``,
so a requeued run matches an uninterrupted one, apart from non-deterministic
cuDNN kernels.

Nothing in here may break training. Every filesystem, git, or CUDA query is
best-effort, and failures are recorded as ``None``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

SEED_HISTORY_FILE = "seed_history.jsonl"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOG_SEED_LINE = re.compile(r"^\[(\S+ \S+)[^\]]*\].*Using seed: (\d+)")
_SLURM_VARS = (
    "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_PARTITION", "SLURMD_NODENAME",
)
# Derived seeds stay below 2**31 so ``seed + rank`` never overflows numpy's
# 32-bit ``np.random.seed`` range under DDP.
_DERIVED_SEED_MOD = 2**31


def derive_epoch_seed(base_seed: int, epoch: int) -> int:
    """Deterministic, well-mixed per-epoch seed from ``(base_seed, epoch)``."""
    import numpy as np

    state = np.random.SeedSequence([int(base_seed), int(epoch)]).generate_state(1, dtype=np.uint64)[0]
    return int(state % _DERIVED_SEED_MOD)


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), *args],
                capture_output=True, text=True, timeout=10, check=True,
            )
            return out.stdout.strip()
        except Exception:
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def collect_environment(device_id: Optional[int] = None) -> dict[str, Any]:
    """Where/what this launch ran on. Best-effort: unknown fields are ``None``."""
    env: dict[str, Any] = {
        "host": platform.node(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "argv": list(sys.argv),
        "git": _git_state(),
        "slurm": {k: os.environ.get(k) for k in _SLURM_VARS if os.environ.get(k)},
        "job_db_model_id": os.environ.get("AIRCC_MODEL_ID"),
    }
    try:
        import numpy
        import timm
        import torch

        env.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "timm": timm.__version__,
            "numpy": numpy.__version__,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        })
        if torch.cuda.is_available():
            dev = torch.cuda.current_device() if device_id is None else int(device_id)
            env["gpu"] = torch.cuda.get_device_name(dev)
    except Exception as exc:  # pragma: no cover - environment-dependent
        env["env_error"] = repr(exc)[:200]
    return env


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _append_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("a") as fh:
        for e in entries:
            fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")


def backfill_from_log(run_dir: os.PathLike | str) -> list[dict]:
    """Seeds of earlier launches, recovered from ``log.txt`` ``Using seed`` lines."""
    log = Path(run_dir) / "log.txt"
    out: list[dict] = []
    try:
        with log.open(errors="replace") as fh:
            for line in fh:
                m = _LOG_SEED_LINE.match(line)
                if m:
                    out.append({"ts": m.group(1), "seed": int(m.group(2)), "source": "backfill:log.txt"})
    except OSError:
        pass
    return out


def prior_launches(run_dir: os.PathLike | str) -> list[dict]:
    """Launch history recorded before this launch.

    Call this *before* this launch logs its own ``Using seed`` line. Otherwise a
    first-time backfill would count the current launch twice. When
    ``seed_history.jsonl`` is missing, the backfilled entries are written to it, so
    the file is complete from then on.
    """
    path = Path(run_dir) / SEED_HISTORY_FILE
    if path.exists():
        return _read_jsonl(path)
    backfilled = backfill_from_log(run_dir)
    if backfilled:
        try:
            _append_jsonl(path, backfilled)
        except OSError:
            pass
    return backfilled


def record_launch(
    run_dir: os.PathLike | str,
    *,
    prior: list[dict],
    seed: int,
    seed_source: str,
    start_epoch: int,
    total_epochs: int,
    epoch_reseed: bool,
    device_id: Optional[int] = None,
) -> list[dict]:
    """Append this launch to ``seed_history.jsonl``; return the full history.

    ``seed_source`` is ``"config"`` (an explicit ``seed=`` from CLI/CSV) or ``"drawn"``
    (``seed`` was null and a random one was drawn). ``trains`` is False for a
    relaunch that only evaluates (start_epoch >= total_epochs, e.g. the sjm-eval lane).
    """
    entry = {
        "ts": _now(),
        "seed": int(seed),
        "source": seed_source,
        "start_epoch": int(start_epoch),
        "total_epochs": int(total_epochs),
        "trains": int(start_epoch) < int(total_epochs),
        "epoch_reseed": bool(epoch_reseed),
        "env": collect_environment(device_id),
    }
    try:
        _append_jsonl(Path(run_dir) / SEED_HISTORY_FILE, [entry])
    except OSError:
        pass
    return list(prior) + [entry]


def checkpoint_seed_history(history: list[dict]) -> list[dict]:
    """Compact history for embedding in checkpoint ``args``.

    The checkpoint keeps seed/epoch/commit/GPU per launch. Full environments stay in
    the jsonl, so dozens of requeues don't bloat every ``.pth.tar``.
    """
    out = []
    for h in history:
        env = h.get("env") or {}
        git = env.get("git") or {}
        out.append({
            "ts": h.get("ts"),
            "seed": h.get("seed"),
            "source": h.get("source"),
            "start_epoch": h.get("start_epoch"),
            "trains": h.get("trains"),
            "epoch_reseed": h.get("epoch_reseed"),
            "git_commit": git.get("commit"),
            "git_dirty": git.get("dirty"),
            "gpu": env.get("gpu"),
            "slurm_job_id": (env.get("slurm") or {}).get("SLURM_JOB_ID"),
        })
    return out
