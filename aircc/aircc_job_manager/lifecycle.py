"""Full per-model lifecycle: build the training command and run it.

A single ``python -m robust_training.adversarial_training`` does **everything** --
train, then (via the config default ``final_eval=True``) the AutoAttack sweep on
best/last/advbest and the comparison plot, all in one process. The in-process
hooks write epoch / heartbeat / best-checkpoint to the AIRCC DB. So the lifecycle
here just builds the command, launches one subprocess, and marks the row
finished/failed.

The command is assembled from the CSV's non-empty override columns
(``csv_spec.build_overrides``) plus dynamically-resolved checkpoint args and
``+machine=aircc``. All continuation jobs init from the dependency's **DB-best**
checkpoint (weights-only continuations via ``continuation.checkpoint_path``;
epoch-continuing resume variants via ``model.resume``).
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

from aircc.aircc_job_manager.csv_spec import build_overrides

REPO_ROOT = Path(__file__).resolve().parents[2]

MEM_FRACTION = "0.47"          # per-process GPU memory cap (2 procs share one B200)
WANDB_PROJECT = "adv_train_aircc"
MACHINE = "aircc"              # +machine=aircc -> dataset.train_dir/eval_dir


def _own_last(models_root: Path, model_name: str) -> Path:
    return models_root / model_name / "last.pth.tar"


def build_command(row: dict, models_root: Path, db, *, python_exe: Optional[str] = None) -> list[str]:
    """Construct the full training command for a CSV row.

    init_mode:
      * scratch      -> model.resume=<own last> (auto-resume if present)
      * continuation -> continuation.checkpoint_path=<dep DB-best> + model.resume=<own last>
                        (own-last wins on restart; else continuation loads weights, epoch->0)
      * resume       -> model.resume=<own last if present else dep DB-best>
                        (inherits epoch+optimizer to continue the counter)
    """
    python_exe = python_exe or sys.executable
    name = row["model_name"]
    init_mode = (row.get("init_mode") or "scratch").strip()
    dep = (row.get("dependency_model_name") or "").strip()
    own_last = _own_last(models_root, name)

    cmd = [python_exe, "-m", "robust_training.adversarial_training"]
    cmd += build_overrides(row)

    if init_mode == "continuation":
        dep_best = _dep_best(db, dep, name)
        cmd.append(f"continuation.checkpoint_path={dep_best}")
        cmd.append(f"model.resume={own_last}")
    elif init_mode == "resume":
        resume_path = own_last if own_last.exists() else _dep_best(db, dep, name)
        cmd.append(f"model.resume={resume_path}")
    else:  # scratch
        cmd.append(f"model.resume={own_last}")

    cmd.append(f"+machine={MACHINE}")
    return cmd


def _dep_best(db, dep: str, name: str) -> str:
    job = db.get(dep) if dep else None
    best = (job.best_checkpoint if job else None) or ""
    if not best:
        raise RuntimeError(f"{name}: dependency '{dep}' has no DB best_checkpoint to init from")
    return best


def run(row: dict, models_root: Path, db, *, val_dir: Optional[str] = None,
        device: str = "cuda", python_exe: Optional[str] = None, log=print) -> bool:
    """Run the full lifecycle (train+eval+plot in one process). True on success."""
    name = row["model_name"]
    try:
        cmd = build_command(row, models_root, db, python_exe=python_exe)
    except Exception as exc:
        log(f"[lifecycle] {name}: cannot build command: {exc}")
        db.mark_failed(name, f"build_command: {exc}")
        return False

    env = dict(os.environ)
    env["AIRCC_MODEL_ID"] = name
    env["AIRCC_THREAT_NORM"] = str(row.get("threat_norm", "") or "")
    env["AIRCC_THREAT_EPS"] = str(row.get("threat_eps", "") or "")
    env["AIRCC_MEM_FRACTION"] = MEM_FRACTION
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["MASTER_PORT"] = str(10000 + random.randint(0, 49999))
    if val_dir:
        env["AIRCC_VAL_DIR"] = val_dir

    log(f"[lifecycle] {name}: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode
    if rc != 0:
        log(f"[lifecycle] {name}: failed (rc={rc})")
        db.mark_failed(name, f"training rc={rc}")
        return False

    db.mark_finished(name)
    log(f"[lifecycle] {name}: FINISHED")
    return True
