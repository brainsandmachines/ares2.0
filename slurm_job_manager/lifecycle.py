"""Full per-model lifecycle: build the training command and run it.

A single ``python -m robust_training.adversarial_training`` does **everything** --
train, then (via the config default ``final_eval=True``) the AutoAttack sweep on
best/last/advbest and the comparison plot, all in one process. The in-process
hooks (``aircc.aircc_job_manager.progress``, already imported by the training
code) write epoch / heartbeat / best-checkpoint to this manager's DB because we
export ``AIRCC_DB`` + ``AIRCC_MODEL_ID``. So the lifecycle here just builds the
command, launches one subprocess, and marks the row finished/failed.

Botero differences vs the aircc lifecycle: **no** ``+machine`` override (the
config default dataset dirs are used), and the CSV's ``training.batch_size`` is
divided by ``SJM_BATCH_DIVISOR`` (rtx6000 = half the 96 GB rtx_pro_6000 batch).
One clean single-GPU process -- no memory fractions, no multi-proc.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Optional

from slurm_job_manager.csv_spec import build_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WANDB_PROJECT = "adv_train_slurm"


def _own_last(models_root: Path, model_name: str) -> Path:
    return models_root / model_name / "last.pth.tar"


def _dep_best(db, dep: str, name: str) -> str:
    job = db.get(dep) if dep else None
    best = (job.best_checkpoint if job else None) or ""
    if not best:
        raise RuntimeError(f"{name}: dependency '{dep}' has no DB best_checkpoint to init from")
    return best


def _apply_batch_divisor(tokens: list[str]) -> list[str]:
    """Divide the CSV's ``training.batch_size`` by ``SJM_BATCH_DIVISOR``.

    The CSV carries the full (rtx_pro_6000, 96 GB) batch size; each partition's
    sbatch sets the divisor to fit its GPU -- ``1`` on rtx_pro_6000 (as-is),
    ``2`` on rtx6000 (half). Unset/1/no-CSV-value is a no-op.
    """
    try:
        div = int(os.environ.get("SJM_BATCH_DIVISOR", "1").strip() or "1")
    except ValueError:
        div = 1
    if div <= 1:
        return tokens
    out = []
    for t in tokens:
        if t.startswith("training.batch_size="):
            try:
                bs = int(t.split("=", 1)[1])
            except ValueError:
                out.append(t)
                continue
            out.append(f"training.batch_size={max(1, bs // div)}")
        else:
            out.append(t)
    return out


def build_command(row: dict, models_root: Path, db, *, python_exe: Optional[str] = None) -> list[str]:
    """Construct the full training command for a CSV row.

    init_mode:
      * scratch      -> model.resume=<own last> (auto-resume if present)
      * continuation -> continuation.checkpoint_path=<dep DB-best> + model.resume=<own last>
      * resume       -> model.resume=<own last if present else dep DB-best>
    """
    python_exe = python_exe or sys.executable
    name = row["model_name"]
    init_mode = (row.get("init_mode") or "scratch").strip()
    dep = (row.get("dependency_model_name") or "").strip()
    own_last = _own_last(models_root, name)

    cmd = [python_exe, "-m", "robust_training.adversarial_training"]
    cmd += _apply_batch_divisor(build_overrides(row))

    if init_mode == "continuation":
        dep_best = _dep_best(db, dep, name)
        cmd.append(f"continuation.checkpoint_path={dep_best}")
        cmd.append(f"model.resume={own_last}")
    elif init_mode == "resume":
        resume_path = own_last if own_last.exists() else _dep_best(db, dep, name)
        cmd.append(f"model.resume={resume_path}")
    else:  # scratch
        cmd.append(f"model.resume={own_last}")

    return cmd


def _lifecycle_env(row: dict, db) -> dict:
    env = dict(os.environ)
    # Point the already-live aircc in-training hooks at THIS manager's DB.
    env["AIRCC_DB"] = db.db_path
    env["AIRCC_MODEL_ID"] = row["model_name"]
    env["AIRCC_THREAT_NORM"] = str(row.get("threat_norm", "") or "")
    env["AIRCC_THREAT_EPS"] = str(row.get("threat_eps", "") or "")
    env["WANDB_PROJECT"] = os.environ.get("SJM_WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    env["MASTER_PORT"] = str(10000 + random.randint(0, 49999))
    # Force the single-process training path (no env:// rendezvous) and stop the
    # orchestrator hooks from firing inside a manager-driven run.
    env["WORLD_SIZE"] = "1"
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    env.pop("ORCH_MODEL_ID", None)
    env.pop("ORCH_DB", None)
    return env


def run(row: dict, models_root: Path, db, *, python_exe: Optional[str] = None,
        tail_lines: int = 400, log=print) -> tuple[int, str]:
    """Run the full lifecycle (train+eval+plot) in one process.

    Streams the child's output to our stdout (the Slurm .out) while keeping a
    bounded tail for the caller's failure classification. Returns
    ``(returncode, log_tail)`` and does **not** touch the DB status -- the
    controller owns the finish/requeue/fail decision so transient errors can be
    requeued rather than hard-failed. A build error returns ``(1, message)``.
    """
    name = row["model_name"]
    try:
        cmd = build_command(row, models_root, db, python_exe=python_exe)
    except Exception as exc:
        msg = f"build_command: {exc}"
        log(f"[lifecycle] {name}: {msg}")
        return 1, msg

    env = _lifecycle_env(row, db)
    log(f"[lifecycle] {name}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    tail: deque[str] = deque(maxlen=tail_lines)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        tail.append(line)
    proc.wait()
    return proc.returncode, "".join(tail)
