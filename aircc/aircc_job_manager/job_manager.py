#!/usr/bin/env python3
"""AIRCC job manager -- one instance per Slurm array task / GPU.

Runs an infinite loop with **2 parallel training lifecycles** on the single
allocated GPU. Each slot atomically claims the next eligible model from the DB,
runs the full lifecycle (train -> eval 3 ckpts -> plot -> mark finished), then
claims again. A main thread requeues stale jobs and heartbeats in-flight models
(covering the eval/plot phases where the training loop is not running).

Safe to start, kill, and restart at any time: claims are atomic, stale running
rows are requeued after 15 min without a heartbeat, and each training resumes
from its own ``last.pth.tar``.

The CSVs are re-read from disk before EVERY claim (never cached for the process
lifetime), and CSV priority/epochs are pushed onto still-pending DB rows first,
so editing a CSV while the manager runs takes effect on every model that has not
been claimed yet -- no need to cancel and resubmit the sbatch.

    # dry-run: show the next eligible commands without launching anything
    python -m aircc.aircc_job_manager.job_manager --dry-run

    # real run (one GPU, 2 slots); env AIRCC_DB must point at the shared sqlite
    AIRCC_DB=/path/aircc_jobs.sqlite python -m aircc.aircc_job_manager.job_manager
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from aircc.aircc_job_manager import lifecycle
from aircc.aircc_job_manager.csv_spec import CSV_DIR, load_spec
from aircc.aircc_job_manager.db import AirccDB

REPO_ROOT = Path(__file__).resolve().parents[2]

IDLE_SLEEP_S = 30      # slot retry interval when no work is claimable
POLL_S = 60            # main-loop requeue + heartbeat cadence
N_SLOTS = 2            # training processes per GPU


def _owner_task() -> int:
    for key in ("SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"):
        v = os.environ.get(key)
        if v:
            base = int(v)
            tid = os.environ.get("SLURM_ARRAY_TASK_ID")
            return base * 100000 + int(tid) if tid else base
    return os.getpid()


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class JobManager:
    def __init__(self, db: AirccDB, csv_dir: Path, models_root: Path, val_dir: str | None):
        self.db = db
        self.csv_dir = csv_dir
        self.models_root = models_root
        self.val_dir = val_dir
        self.owner = _owner_task()
        self.slots: list[str | None] = [None] * N_SLOTS
        self.lock = threading.Lock()
        self.stop = threading.Event()

    # ---- spec: re-read from disk on every claim ---------------------------
    def _load_spec(self) -> tuple[dict, dict]:
        """Fresh ``(rows, deps)`` off disk -- no snapshot is ever cached.

        Read before every claim so a CSV edit reaches every model that has not
        been claimed yet, without cancelling and resubmitting the sbatch. Models
        already training keep the row they were launched with.
        """
        _, rows, deps = load_spec(self.csv_dir)
        return rows, deps

    # ---- worker: claim + run lifecycle ------------------------------------
    def _worker(self, idx: int) -> None:
        while not self.stop.is_set():
            # Re-read the CSVs, then push priority/epochs onto still-pending rows,
            # THEN claim -- so this claim sees the CSV as it is on disk right now.
            try:
                rows, deps = self._load_spec()
            except Exception as exc:
                # Loud and never stale: a malformed/half-written CSV blocks claiming
                # until it parses again, rather than silently reusing older values.
                _log(f"slot{idx} CSV RELOAD FAILED, not claiming: {exc}")
                self.stop.wait(IDLE_SLEEP_S)
                continue
            try:
                changed = self.db.sync_pending_spec(
                    {n: (int(r["training.epochs"]), int(r["priority"])) for n, r in rows.items()}
                )
                if changed:
                    _log(f"slot{idx} csv sync: {len(changed)} pending row(s) updated "
                         f"({', '.join(changed[:5])}{' ...' if len(changed) > 5 else ''})")
            except Exception as exc:  # pragma: no cover - defensive
                _log(f"slot{idx} csv sync error (ignored): {exc}")
                self.stop.wait(IDLE_SLEEP_S)
                continue
            try:
                job = self.db.claim_next(self.owner, deps)
            except Exception as exc:  # pragma: no cover - defensive
                _log(f"slot{idx} claim_next error (ignored): {exc}")
                self.stop.wait(IDLE_SLEEP_S)
                continue
            if job is None:
                self.stop.wait(IDLE_SLEEP_S)
                continue
            row = rows.get(job.model_name)
            if row is None:
                # Seeded model with no CSV row (shouldn't happen); fail it loudly.
                self.db.mark_failed(job.model_name, "no CSV row for seeded model")
                continue
            with self.lock:
                self.slots[idx] = job.model_name
            _log(f"slot{idx} claimed {job.model_name}")
            try:
                lifecycle.run(row, self.models_root, self.db,
                              val_dir=self.val_dir, log=_log)
            except Exception as exc:  # pragma: no cover - defensive
                _log(f"slot{idx} lifecycle crashed for {job.model_name}: {exc}")
                self.db.mark_failed(job.model_name, f"lifecycle crash: {exc}")
            finally:
                with self.lock:
                    self.slots[idx] = None

    # ---- main loop: requeue stale + heartbeat in-flight -------------------
    def run(self) -> None:
        _log(f"job manager up: owner={self.owner} slots={N_SLOTS} models_root={self.models_root}")
        # Requeue stale rows BEFORE any worker claims, so a claim never precedes a
        # requeue (ordering: upsert[seed] -> requeue -> claim).
        try:
            n = self.db.requeue_stale()
            if n:
                _log(f"startup requeued {n} stale job(s)")
        except Exception as exc:  # pragma: no cover - defensive
            _log(f"startup requeue error (ignored): {exc}")
        workers = [threading.Thread(target=self._worker, args=(i,), daemon=True)
                   for i in range(N_SLOTS)]
        for w in workers:
            w.start()
        try:
            while not self.stop.is_set():
                try:
                    n = self.db.requeue_stale()
                    if n:
                        _log(f"requeued {n} stale job(s)")
                    with self.lock:
                        active = [s for s in self.slots if s]
                    for name in active:
                        self.db.heartbeat(name)
                except Exception as exc:  # pragma: no cover - defensive
                    _log(f"main-loop error (ignored): {exc}")
                self.stop.wait(POLL_S)
        except KeyboardInterrupt:
            _log("interrupted; shutting down")
            self.stop.set()

    # ---- dry-run ----------------------------------------------------------
    def dry_run(self, n: int) -> None:
        rows, deps = self._load_spec()
        # Mirror claim_next's filter exactly: parked rows (owner_task=-1, still
        # 'pending') are never claimable, so they must not show up here either.
        pending = sorted((j for j in self.db.all_jobs()
                          if j.status == "pending" and j.owner_task is None),
                         key=lambda j: j.priority)
        shown = 0
        _log(f"dry-run: next {n} eligible job(s) in claim order")
        for j in pending:
            dep = deps.get(j.model_name, "")
            if dep:
                dj = self.db.get(dep)
                if not (dj and dj.status == "finished" and (dj.best_checkpoint or "").strip()):
                    continue
            row = rows.get(j.model_name)
            if row is None:
                continue
            try:
                cmd = lifecycle.build_command(row, self.models_root, self.db)
                print(f"\n# [{shown}] {j.model_name} (priority={j.priority}, init_mode={row['init_mode']})")
                print(" ".join(cmd))
            except Exception as exc:
                print(f"\n# [{shown}] {j.model_name}: build error: {exc}")
            shown += 1
            if shown >= n:
                break
        if shown == 0:
            print("# (no eligible pending jobs)")


def main() -> int:
    ap = argparse.ArgumentParser(prog="aircc.job_manager")
    ap.add_argument("--db", default=os.environ.get("AIRCC_DB"),
                    help="SQLite DB path (default: $AIRCC_DB).")
    ap.add_argument("--models-root", type=Path,
                    default=Path(os.environ.get("MODELS_ROOT", REPO_ROOT / "results" / "models")))
    ap.add_argument("--val-dir", default=os.environ.get("AIRCC_VAL_DIR"))
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-run-count", type=int, default=8)
    args = ap.parse_args()

    if not args.db:
        ap.error("no DB path (set --db or $AIRCC_DB)")

    # Fail fast on a bad --csv-dir; the workers re-read (and re-validate) the CSVs
    # before every claim, so nothing is cached from here.
    all_rows, _, _ = load_spec(args.csv_dir)
    _log(f"csv spec ok: {len(all_rows)} row(s) in {args.csv_dir}")
    db = AirccDB(args.db)
    mgr = JobManager(db, args.csv_dir, args.models_root, args.val_dir)

    if args.dry_run:
        mgr.dry_run(args.dry_run_count)
        return 0

    # AIRCC_DB must be exported so the in-training hooks write to this same DB.
    os.environ["AIRCC_DB"] = args.db
    mgr.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
