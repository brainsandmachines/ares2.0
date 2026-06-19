"""Persistent botero daemon: poll squeue, top up GPUs, reconcile failures.

Runs continuously with a configurable interval (``ORCH_INTERVAL_MINUTES``). A
lockfile guarantees a single instance; the 20-min cron watchdog restarts it if
it dies. The loop is the immediate-relaunch mechanism: when a job ends and frees
a GPU, the next poll tops the slot up (≤ one interval of latency).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

from .config import Config
from .core import Orchestrator, SlurmClient
from .db import OrchestratorDB, STAGE_COMPLETED
from .failure_analyzer import analyze_failure, extract_traceback
from . import notify

logger = logging.getLogger("orchestrator.daemon")

LOCKFILE = "/tmp/ares_orchestrator.lock"


def _acquire_lock() -> bool:
    """Single-instance guard. Returns True if we got the lock."""
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE) as fh:
                pid = int(fh.read().strip() or "0")
            os.kill(pid, 0)  # raises if not running
            return False  # another live instance holds it
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale lock; reclaim it
    with open(LOCKFILE, "w") as fh:
        fh.write(str(os.getpid()))
    return True


def _release_lock() -> None:
    try:
        os.remove(LOCKFILE)
    except FileNotFoundError:
        pass


def reconcile(orch: Orchestrator, cfg: Config) -> None:
    """Detect finished/failed jobs among RUNNING rows and act deterministically.

    * job still active                          -> leave it
    * job gone, stage == COMPLETED              -> success, nothing to do
    * job gone, stage != COMPLETED, traceback   -> FAILED + MD5-fenced analyzer
    * job gone, stage != COMPLETED, no traceback -> requeue PENDING (resume)
    """
    for row in orch.db.list_models(status="RUNNING"):
        if not row.slurm_job_id:
            continue
        if orch.slurm.is_active(row.slurm_job_id):
            continue
        fresh = orch.db.get_model_state(row.model_id)
        if fresh is None or fresh.current_stage == STAGE_COMPLETED:
            continue
        log_text = orch.slurm.read_log(row.slurm_job_id)
        tb = extract_traceback(log_text) if log_text else ""
        if "Traceback (most recent call last):" in tb:
            if cfg.enable_failure_analyzer:
                analyze_failure(
                    orch.db,
                    row.model_id,
                    log_text,
                    cfg.recommendations_dir,
                    llm_client=notify.make_llm_client(),
                    emailer=notify.make_emailer(cfg),
                )
            else:
                orch.db.mark_failed(row.model_id)
        else:
            # clean exit without a traceback (timelimit/preemption) -> resume
            orch.db.set_status(row.model_id, "PENDING")
            logger.info("requeued %s (no traceback; likely timeout)", row.model_id)


def run_once(cfg: Optional[Config] = None, dry_run: bool = False) -> None:
    cfg = cfg or Config.load()
    db = OrchestratorDB(cfg.db_path)
    slurm = SlurmClient(cfg)
    orch = Orchestrator(db, slurm, cfg)
    reconcile(orch, cfg)
    orch.requeue_stale()
    orch.tick(dry_run=dry_run)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    cfg = Config.load()
    if not _acquire_lock():
        logger.info("another orchestrator instance is running; exiting")
        return
    logger.info("orchestrator daemon up; interval=%.1f min", cfg.interval_minutes)
    try:
        while True:
            try:
                run_once(cfg)
            except Exception:
                logger.exception("tick failed; continuing")
            time.sleep(cfg.interval_minutes * 60)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
