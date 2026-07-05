"""Cluster-side array worker: requeue dead owners -> claim one model -> run it.

Every task of a manager array (``slurm/manager_<partition>.sbatch``) runs this
once and exits. One task = one GPU = one model:

1. release rows whose owning Slurm task is dead (liveness, not heartbeat),
2. atomically claim the single top-priority eligible row (test lane first),
3. run the full single-process lifecycle (train -> final_eval AA -> plot),
4. success -> mark_finished; transient error -> requeue; real error -> mark
   FAILED with the log tail + signature (the botero monitor escalates unknown
   signatures to codex).

The claimed model id is written to ``SJM_TRAP_FILE`` (if set) so the sbatch's
SIGTERM trap can ``release`` the row on a Slurm time-limit kill.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from .classify import ACTION_REQUEUE, classify, error_hash
from .csv_spec import CSV_DIR, deps_map, load_all_rows, row_map
from .db import JobDB
from . import lifecycle

logger = logging.getLogger("sjm.controller")

_ACTIVE_SLURM_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "RESIZING"}


def _in_squeue(slurm_job_id: int) -> bool:
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
    """True if the owning Slurm job is still alive (present in squeue or sacct)."""
    return _in_squeue(slurm_job_id) or _sacct_active(slurm_job_id)


def _write_trap_file(model_name: str) -> None:
    path = os.environ.get("SJM_TRAP_FILE")
    if not path:
        return
    try:
        Path(path).write_text(model_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not write trap file %s: %s", path, exc)


def run_once(db: JobDB, rows: dict, deps: dict, models_root: Path,
             slurm_job_id: int) -> int:
    """Do one requeue+claim+run cycle. Returns a process exit code."""
    freed = db.requeue_dead(_default_slurm_active)
    if freed:
        logger.info("released %d dead-owner row(s)", freed)

    job = db.claim_next(slurm_job_id, deps)
    if job is None:
        logger.info("no claimable work; exiting 0")
        return 0

    row = rows.get(job.model_name)
    if row is None:
        db.mark_failed(job.model_name, "no CSV row for seeded model")
        logger.error("claimed %s has no CSV row -> FAILED", job.model_name)
        return 1

    _write_trap_file(job.model_name)
    logger.info("claimed %s (test=%s, priority=%s)", job.model_name, job.is_test, job.priority)

    rc, tail = lifecycle.run(row, models_root, db, log=lambda m: print(m, flush=True))

    fresh = db.get(job.model_name)
    if rc == 0 and fresh is not None and fresh.status == "running":
        db.mark_finished(job.model_name)
        logger.info("%s FINISHED", job.model_name)
        return 0

    # Non-zero (or the row was released mid-run by the trap): classify the tail.
    if fresh is not None and fresh.status == "pending":
        # Released by the SIGTERM trap already; leave it claimable.
        logger.info("%s was released (time-limit); leaving claimable", job.model_name)
        return rc
    action = classify(tail)
    if action == ACTION_REQUEUE:
        db.release(job.model_name)
        logger.warning("%s transient (rc=%s) -> requeued", job.model_name, rc)
        return rc
    ehash = error_hash(tail)
    db.mark_failed(job.model_name, error=tail, error_hash=ehash)
    logger.error("%s FAILED (rc=%s, %s) [%s]", job.model_name, rc, action, ehash)
    return 1 if rc == 0 else rc


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="slurm_job_manager.controller")
    ap.add_argument("--db", default=os.environ.get("SJM_DB"))
    ap.add_argument("--models-root", type=Path,
                    default=Path(os.environ.get("MODELS_ROOT",
                                                lifecycle.REPO_ROOT / "results" / "models")))
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the next eligible command(s) without claiming/launching.")
    ap.add_argument("--dry-run-count", type=int, default=8)
    args = ap.parse_args()

    if not args.db:
        ap.error("no DB path (set --db or $SJM_DB)")

    all_rows = load_all_rows(args.csv_dir)
    rows = row_map(all_rows)
    deps = deps_map(all_rows)
    db = JobDB(args.db)

    if args.dry_run:
        return _dry_run(db, rows, deps, args.models_root, args.dry_run_count)

    slurm_job_id = int(os.environ.get("SLURM_JOB_ID", os.getpid()))
    return run_once(db, rows, deps, args.models_root, slurm_job_id)


def _dry_run(db: JobDB, rows: dict, deps: dict, models_root: Path, n: int) -> int:
    pending = [j for j in db.all_jobs() if j.status == "pending"]
    shown = 0
    print(f"# dry-run: next {n} eligible job(s) in claim order (test lane first)")
    for j in pending:
        dep = deps.get(j.model_name, "")
        if dep:
            dj = db.get(dep)
            if not (dj and dj.status == "finished" and (dj.best_checkpoint or "").strip()):
                continue
        row = rows.get(j.model_name)
        if row is None:
            continue
        try:
            cmd = lifecycle.build_command(row, models_root, db)
            print(f"\n# [{shown}] {j.model_name} (test={j.is_test}, priority={j.priority})")
            print(" ".join(cmd))
        except Exception as exc:
            print(f"\n# [{shown}] {j.model_name}: build error: {exc}")
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print("# (no eligible pending jobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
