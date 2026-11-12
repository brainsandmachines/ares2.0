"""Launch eval-lane jobs: one ``sjm-eval`` Slurm job on ``main`` per trained model.

A row reaches status ``eval`` when the training lane finishes its last epoch
(``JobDB.mark_trained``), when a fully-trained run is released (trap / dead
owner), or when ``reopen_eval`` re-opens a finished model with missing AA cells.
Each ``sjm-eval`` job (``slurm/manager_eval_main.sbatch``) runs the controller in
the eval lane: it claims ONE ``eval`` row, runs the full AutoAttack sweep +
shape-bias, and the in-process hook writes ``best_checkpoint``/``best_score``.

:func:`top_up` keeps exactly enough jobs queued: it submits
``unowned eval rows - PENDING sjm-eval jobs``. It is called right after a
training run ends (from the compute node), by ``reopen_eval --apply``, and by
the hourly ``cron_release`` sweep, which catches any submit that failed. A
surplus job (two callers racing) finds nothing to claim and exits in seconds.
"""

from __future__ import annotations

import getpass
import logging
import os
import subprocess
from typing import Callable, Optional

from .lifecycle import REPO_ROOT

logger = logging.getLogger("sjm.eval_lane")

EVAL_JOB_NAME = "sjm-eval"
EVAL_SBATCH = REPO_ROOT / "slurm_job_manager" / "slurm" / "manager_eval_main.sbatch"


def pending_eval_jobs(run: Callable = subprocess.run) -> Optional[int]:
    """How many ``sjm-eval`` jobs are waiting to start (None if squeue failed)."""
    user = os.environ.get("USER") or getpass.getuser()
    try:
        cp = run(["squeue", "-u", user, "-n", EVAL_JOB_NAME, "-t", "PENDING", "-h", "-o", "%i"],
                 capture_output=True, text=True, timeout=60)
    except Exception as exc:
        logger.warning("squeue failed: %s", exc)
        return None
    if cp.returncode != 0:
        logger.warning("squeue rc=%s: %s", cp.returncode, (cp.stderr or "").strip())
        return None
    return sum(1 for line in cp.stdout.splitlines() if line.strip())


def submit(n: int, run: Callable = subprocess.run) -> list[str]:
    """sbatch ``n`` eval jobs; return the job ids that were accepted."""
    ids: list[str] = []
    for _ in range(max(0, n)):
        try:
            cp = run(["sbatch", "--parsable", str(EVAL_SBATCH)], cwd=str(REPO_ROOT),
                     capture_output=True, text=True, timeout=120)
        except Exception as exc:
            logger.warning("sbatch failed: %s", exc)
            break
        if cp.returncode != 0:
            logger.warning("sbatch rc=%s: %s", cp.returncode, (cp.stderr or "").strip())
            break
        ids.append(cp.stdout.strip().split(";")[0])
    return ids


def top_up(db, dry_run: bool = False, run: Callable = subprocess.run) -> dict:
    """Promote stray fully-trained pending rows to ``eval``, then queue enough eval jobs.

    Returns ``{"promoted": [...], "unowned": N, "pending_jobs": M, "submitted": [...]}``.
    Never raises: a failed squeue submits nothing (the next sweep retries).
    """
    report = {"promoted": db.promote_trained(dry_run=dry_run), "unowned": 0,
              "pending_jobs": None, "submitted": []}
    unowned = db.unowned_eval()
    if dry_run:
        # promote_trained(dry_run) only listed them; count them as if promoted.
        report["unowned"] = len(unowned) + len(report["promoted"])
    else:
        report["unowned"] = len(unowned)
    if report["unowned"] == 0:
        return report
    pending = pending_eval_jobs(run=run)
    report["pending_jobs"] = pending
    if pending is None:
        return report
    need = report["unowned"] - pending
    if need <= 0:
        return report
    if dry_run:
        report["submitted"] = ["(dry-run)"] * need
    else:
        report["submitted"] = submit(need, run=run)
    return report
