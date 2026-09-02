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

import json
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

# init_mode="resume" rows whose CSV row carries a non-empty resume_offset_assumed
# are "shift-managed": training.epochs and the epsilon ramp boundaries are
# re-derived from the checkpoint the run ACTUALLY resumes from (see
# _resolve_resume_target below), so these 5 columns must never be emitted twice.
RESUME_SHIFT_COLUMNS = (
    "training.epochs",
    "epsilon_schedule.warmup_epochs",
    "epsilon_schedule.ramp_start_epoch",
    "epsilon_schedule.ramp_end_epoch",
    "epsilon_schedule.fixed_start_epoch",
)


def _own_last(models_root: Path, model_name: str) -> Path:
    return models_root / model_name / "last.pth.tar"


def _resume_target_file(models_root: Path, model_name: str) -> Path:
    # Deliberately the same sidecar name aircc writes, so runs migrated off the
    # AIRCC cluster reuse the target they already resolved there.
    return models_root / model_name / ".aircc_resume_target.json"


def _peek_resume_epoch(checkpoint_path) -> Optional[int]:
    """Best-effort read of the epoch a timm-style checkpoint will resume at.

    Mirrors ``timm.models.resume_checkpoint``'s epoch extraction exactly (incl.
    the ``version>1`` +1 adjustment) without touching a model, so we can decide
    the resume target *before* launching training. Returns None on any failure
    (missing file, unreadable checkpoint, no 'epoch' key) -- callers must treat
    that as "unknown, don't shift".
    """
    p = Path(checkpoint_path)
    if not p.is_file():
        return None
    try:
        import torch  # heavy, import lazily

        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict) or "epoch" not in ckpt:
        return None
    try:
        epoch = int(ckpt["epoch"])
    except (TypeError, ValueError):
        return None
    if int(ckpt.get("version", 1) or 1) > 1:
        epoch += 1  # old checkpoints incremented before save; match timm's convention
    return epoch


def _sync_db_total_epochs(db, name: str, target: int) -> None:
    if db is None:
        return
    try:
        db.reconcile(name, target)
    except Exception:
        pass  # cosmetic only -- never let a dashboard-sync failure break the launch


def _resolve_resume_target(row: dict, resume_path, own_last: Path, models_root: Path,
                           name: str, db=None) -> Optional[int]:
    """Return the total-epochs target a shift-managed resume row should train to.

    Computed ONCE, the first time this model actually launches (from the dep's
    checkpoint), as ``peeked_start_epoch + (csv_training_epochs - resume_offset_assumed)``
    -- i.e. "whatever epoch we really start at, plus the originally-intended
    number of new epochs". Persisted next to the model's own checkpoints so
    every subsequent restart (which resumes from own_last, already past that
    start) reuses the SAME target instead of re-deriving it from a now-later
    epoch, which would silently shrink the remaining training length on every
    restart. Returns None if the peek fails (e.g. dep checkpoint has no 'epoch'
    key) -- caller falls back to the static CSV values.

    Also mirrors the resolved target into the DB's total_epochs (best-effort,
    via db.reconcile) purely so dashboards show the true denominator.
    """
    target_file = _resume_target_file(models_root, name)
    if own_last.exists() and target_file.exists():
        try:
            target = int(json.loads(target_file.read_text())["target_epochs"])
        except Exception:
            target = None  # corrupt/unreadable sidecar -- fall through and recompute
        if target is not None:
            _sync_db_total_epochs(db, name, target)
            return target

    start_epoch = _peek_resume_epoch(resume_path)
    if start_epoch is None:
        return None
    offset = int(row["resume_offset_assumed"])
    increment = int(row["training.epochs"]) - offset
    target = start_epoch + increment
    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(json.dumps({
            "target_epochs": target, "start_epoch": start_epoch,
            "resume_offset_assumed": offset, "resume_path": str(resume_path),
        }))
    except OSError:
        pass  # best-effort; a missing sidecar just means the next restart re-peeks
    _sync_db_total_epochs(db, name, target)
    return target


def _shifted_resume_overrides(row: dict, target: Optional[int]) -> list[str]:
    """Re-emit RESUME_SHIFT_COLUMNS shifted by (target - csv training.epochs).

    shift=0 (target is None, or exactly matches the CSV's offset assumption)
    reproduces the static CSV values untouched.
    """
    shift = 0 if target is None else target - int(row["training.epochs"])
    out: list[str] = []
    for col in RESUME_SHIFT_COLUMNS:
        val = str(row.get(col, "")).strip()
        if val == "":
            continue
        out.append(f"{col}={int(val) + shift}")
    return out


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
                        (inherits epoch+optimizer to continue the counter). If the row
                        has resume_offset_assumed set, training.epochs + the epsilon
                        ramp boundaries are shifted to match the ACTUAL resumed epoch
                        (see _resolve_resume_target) instead of trusting that the dep's
                        DB-best checkpoint is at its final epoch.
    """
    python_exe = python_exe or sys.executable
    name = row["model_name"]
    init_mode = (row.get("init_mode") or "scratch").strip()
    dep = (row.get("dependency_model_name") or "").strip()
    own_last = _own_last(models_root, name)
    shift_managed = init_mode == "resume" and bool((row.get("resume_offset_assumed") or "").strip())

    overrides = build_overrides(row, skip=RESUME_SHIFT_COLUMNS if shift_managed else None)

    if init_mode == "continuation":
        dep_best = _dep_best(db, dep, name)
        overrides.append(f"continuation.checkpoint_path={dep_best}")
        overrides.append(f"model.resume={own_last}")
    elif init_mode == "resume":
        resume_path = own_last if own_last.exists() else _dep_best(db, dep, name)
        overrides.append(f"model.resume={resume_path}")
        if shift_managed:
            target = _resolve_resume_target(row, resume_path, own_last, models_root, name, db=db)
            if target is None:
                # Falling back to the static CSV epochs is a real protocol change
                # (on aircc these rows shifted by up to 51 epochs), and the usual
                # cause is an unreadable checkpoint or a torch-less interpreter --
                # both invisible otherwise. Never let that pass silently.
                print(f"[lifecycle] WARNING {name}: could not peek the resume epoch from "
                      f"{resume_path}; using the static CSV training.epochs="
                      f"{row.get('training.epochs')} instead of a shifted target",
                      file=sys.stderr)
            overrides += _shifted_resume_overrides(row, target)
    else:  # scratch
        overrides.append(f"model.resume={own_last}")

    return [python_exe, "-m", "robust_training.adversarial_training"] + _apply_batch_divisor(overrides)


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
