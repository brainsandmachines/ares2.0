"""Cluster-side array worker: requeue stale -> claim one job -> run its stages.

Every task of the controller array (``sbatches/controller.sbatch``) runs this
once and exits. The pull model lives here:

1. release stale/dead owners so their rows become claimable again,
2. atomically claim the single highest-priority eligible row for this partition
   (PLOTTING > AA_EVAL > TRAINING > PENDING),
3. run the remaining pipeline for that row, in-process, advancing status via the
   in-job hooks (training -> AA_EVAL, AA -> PLOTTING, plotting -> FINISHED),
4. on failure, classify deterministically and either requeue or mark FAILED.

Stage *execution* is delegated: train/AA reuse the proven launcher command
builder (``sbatches/run_stage.sh`` -> ``parse_train_job``); plotting calls the
orchestrator plotting script. Both are injected as ``run_stage`` so the
orchestration logic is fully testable offline (no Slurm, no GPU).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from .classify import (
    ACTION_REQUEUE,
    classify,
    error_hash,
)
from .db import (
    ModelRow,
    OrchestratorDB,
    STATUS_AA_EVAL,
    STATUS_FINISHED,
    STATUS_PLOTTING,
    STATUS_TRAINING,
)

logger = logging.getLogger("orchestrator.controller")

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_STAGE_SH = REPO_ROOT / "sbatches" / "run_stage.sh"
PLOT_SCRIPT = REPO_ROOT / "data_analysis" / "plot_autoattack_comparation_orch.py"

# Per-status staleness thresholds (seconds); overridable via .env-style env vars.
_STALE_ENV = {
    STATUS_TRAINING: ("ORCH_TRAIN_STALE_HOURS", 8.0),
    STATUS_AA_EVAL: ("ORCH_AA_STALE_HOURS", 36.0),
    STATUS_PLOTTING: ("ORCH_PLOT_STALE_HOURS", 2.0),
}

# stage -> the status the in-job hook must reach for the stage to count as done.
_EXPECTED_AFTER = {
    "train": STATUS_AA_EVAL,
    "aa": STATUS_PLOTTING,
    "plot": STATUS_FINISHED,
}

# Run plan keyed by the claimed row's entry status.
_PLAN = {
    STATUS_TRAINING: ("train", "aa", "plot"),
    STATUS_AA_EVAL: ("aa", "plot"),
    STATUS_PLOTTING: ("plot",),
}


class StageFailure(Exception):
    """A stage exited non-zero or failed to advance the row's status."""

    def __init__(self, stage: str, log_text: str):
        super().__init__(f"stage {stage} failed")
        self.stage = stage
        self.log_text = log_text


def _thresholds() -> dict[str, int]:
    out: dict[str, int] = {}
    for status, (env, default_hours) in _STALE_ENV.items():
        hours = float(os.environ.get(env, default_hours))
        out[status] = int(hours * 3600)
    return out


# --------------------------------------------------------------------------
# Default (real) implementations of the injected helpers.
# --------------------------------------------------------------------------
_ACTIVE_SLURM_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "RESIZING"}


def _in_squeue(slurm_job_id: int) -> bool:
    """True if the job currently appears in squeue (definitively alive)."""
    try:
        out = subprocess.run(
            ["squeue", "-j", str(slurm_job_id), "-h", "-o", "%i"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return False
    return bool(out.strip())


def _sacct_active(slurm_job_id: int) -> bool:
    try:
        out = subprocess.run(
            ["sacct", "-j", str(slurm_job_id), "-n", "-X", "-o", "State"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return False
    states = [s.strip().split()[0] for s in out.splitlines() if s.strip()]
    return any(s.upper() in _ACTIVE_SLURM_STATES for s in states)


def _default_slurm_active(slurm_job_id: int) -> bool:
    """True if the owning Slurm job is still alive.

    A job is alive if it is present in ``squeue`` OR ``sacct`` reports it in an
    active state. Requeue therefore requires BOTH the staleness threshold to be
    exceeded AND the job to be absent from squeue (and not sacct-active) -- so a
    long-but-legitimately-running task is never yanked out from under itself.
    """
    return _in_squeue(slurm_job_id) or _sacct_active(slurm_job_id)


def _run_streaming(cmd: list[str], env: dict, tail_lines: int = 400) -> tuple[int, str]:
    """Run ``cmd``, streaming output to our stdout (the Slurm .out) while keeping
    a bounded tail for failure classification."""
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    tail: deque[str] = deque(maxlen=tail_lines)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        tail.append(line)
    proc.wait()
    return proc.returncode, "".join(tail)


def _default_run_stage(stage: str, job: ModelRow) -> tuple[int, str]:
    """Real stage execution: bash launcher for train/AA, python for plotting."""
    env = os.environ.copy()
    env["ORCH_MODEL_ID"] = job.model_id
    if stage in ("train", "aa"):
        env["SLURM_JOB_NAME"] = job.job_name or job.model_id
        env["ORCH_TARGET_EPOCH"] = str(job.target_epoch)
        env["ORCH_STAGE_MODE"] = stage
        return _run_streaming(["bash", str(RUN_STAGE_SH), stage], env)
    # plot
    model_dir = job.model_dir or job.model_id
    mname = os.path.basename(model_dir)
    mroot = os.path.dirname(model_dir)
    cmd = [
        sys.executable, str(PLOT_SCRIPT),
        "--plot_name", mname, "--models", mname,
        "--checkpoint-kind", "all",
        "--models-root", mroot, "--out-dir", model_dir,
    ]
    return _run_streaming(cmd, env)


# --------------------------------------------------------------------------
# Orchestration (the testable core).
# --------------------------------------------------------------------------
def requeue_stale(
    db: OrchestratorDB,
    slurm_active: Callable[[int], bool] = _default_slurm_active,
) -> int:
    """Release rows whose owning Slurm task is dead AND idle past its threshold."""
    n = 0
    for row in db.find_stale_candidates(_thresholds()):
        if row.slurm_job_id is None:
            continue
        if not slurm_active(row.slurm_job_id):
            db.requeue(row.model_id)
            logger.info(
                "requeued stale %s (status=%s, dead job %s)",
                row.model_id, row.status, row.slurm_job_id,
            )
            n += 1
    return n


def run_job(
    db: OrchestratorDB,
    job: ModelRow,
    run_stage: Callable[[str, ModelRow], tuple[int, str]] = _default_run_stage,
) -> int:
    """Run the remaining pipeline for a claimed ``job``. Returns a process exit code."""
    plan = _PLAN.get(job.status)
    if plan is None:
        logger.error("claimed %s in unexpected status %s", job.model_id, job.status)
        db.mark_failed(job.model_id)
        return 1
    try:
        for stage in plan:
            rc, log = run_stage(stage, job)
            fresh = db.get_model_state(job.model_id)
            expected = _EXPECTED_AFTER[stage]
            if rc != 0 or fresh is None or fresh.status != expected:
                raise StageFailure(stage, log)
            logger.info("%s: stage %s -> %s", job.model_id, stage, expected)
        return 0
    except StageFailure as exc:
        action = classify(exc.log_text)
        ehash = error_hash(exc.log_text)
        if action == ACTION_REQUEUE:
            db.requeue(job.model_id)
            logger.warning("%s: stage %s transient -> requeued", job.model_id, exc.stage)
        else:
            # FAIL or UNKNOWN: park as FAILED with a signature; the botero monitor
            # re-reads the Slurm log and escalates UNKNOWN signatures to codex.
            db.mark_failed(job.model_id, ehash)
            logger.error(
                "%s: stage %s failed (%s) -> FAILED [%s]",
                job.model_id, exc.stage, action, ehash,
            )
        return 1


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(prog="orchestrator.controller")
    ap.add_argument("--partition", default=os.environ.get("SLURM_JOB_PARTITION", ""))
    args = ap.parse_args()

    db_path = os.environ.get("ORCH_DB")
    if not db_path:
        logger.error("ORCH_DB unset; cannot run controller")
        return 2
    if not args.partition:
        logger.error("no --partition / SLURM_JOB_PARTITION; cannot claim")
        return 2
    # The per-array-task id; stamped as the row owner so the monitor can map a
    # failed Slurm task back to its DB row.
    slurm_job_id = int(os.environ.get("SLURM_JOB_ID", os.getpid()))

    db = OrchestratorDB(db_path)
    requeue_stale(db)
    job = db.claim_next(args.partition, slurm_job_id)
    if job is None:
        logger.info("no claimable work for %s; exiting 0", args.partition)
        return 0
    logger.info(
        "claimed %s (status=%s, plan=%s) on %s",
        job.model_id, job.status, _PLAN.get(job.status), args.partition,
    )
    return run_job(db, job)


if __name__ == "__main__":
    raise SystemExit(main())
