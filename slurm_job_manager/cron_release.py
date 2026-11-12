#!/usr/bin/env python3
"""Hourly sjm sweep: dead/stalled owners, eval handoff, eval-lane top-up.

1. ``JobDB.cron_release``: release rows no live task owns, scancel stalled owners,
   and hand off old-config runs stuck in in-process final_eval on a training
   partition (scancel -> the trap releases the row to ``eval``).
2. ``eval_lane.top_up``: move stray fully-trained ``pending`` rows to ``eval`` and
   sbatch enough ``sjm-eval`` jobs on ``main`` for every unowned ``eval`` row --
   the catch-up for a submit that failed on the compute node.

Runs on the cluster login node, driven by ``scripts/cron_release.sh`` from a Botero cron::

    python -m slurm_job_manager.cron_release --db $SJM_DB [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional

from . import eval_lane
from .db import JobDB

# Epoch-second timestamps: local wall-clock times are ambiguous across a DST change.
_SLURM_ENV = {**os.environ, "SLURM_TIME_FORMAT": "%s"}

# Longest self-recovering in-run silence in 2026 logs is ~2.3h; Dec 2025 had 3 runs
# silent 7.6h at once and 9 swin_b stalled together on 2026-09-01, so >2 = shared cause.
DEFAULT_STALL_S = 6 * 3600
DEFAULT_MAX_STALLED = 2


def _run(argv: list[str]) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60, env=_SLURM_ENV)
    except Exception:
        return None


def _parse(out: str) -> tuple[str, Optional[int]]:
    state, _, start = out.strip().splitlines()[0].partition("|")
    start = start.strip()
    return state.split()[0].upper(), (int(start) if start.isdigit() else None)


def owner_state(job_id: int) -> Optional[tuple[str, Optional[int]]]:
    """``(state, start_ts)``; ``("", None)`` once Slurm has forgotten the id; None if unknown."""
    sq = _run(["squeue", "-j", str(job_id), "-h", "-o", "%T|%S"])
    if sq is not None and sq.returncode == 0 and sq.stdout.strip():
        return _parse(sq.stdout)
    sa = _run(["sacct", "-j", str(job_id), "-X", "-n", "-P", "-o", "State,Start"])
    if sa is None or sa.returncode != 0:
        return None
    if sa.stdout.strip():
        return _parse(sa.stdout)
    if sq is not None and "Invalid job id" in sq.stderr:
        return ("", None)
    return None


def partition_of(job_id: int) -> Optional[str]:
    """The partition a live job runs on, or None if squeue did not say."""
    sq = _run(["squeue", "-j", str(job_id), "-h", "-o", "%P"])
    if sq is None or sq.returncode != 0 or not sq.stdout.strip():
        return None
    return sq.stdout.strip().splitlines()[0].strip()


def cancel(job_id: int) -> bool:
    r = _run(["scancel", str(job_id)])
    return r is not None and r.returncode == 0


def _age(s: int) -> str:
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def main() -> int:
    ap = argparse.ArgumentParser(prog="slurm_job_manager.cron_release")
    ap.add_argument("--db", default=os.environ.get("SJM_DB"))
    ap.add_argument("--stall-s", type=int, default=DEFAULT_STALL_S,
                    help="scancel a RUNNING owner whose heartbeat is older than this")
    ap.add_argument("--max-stalled", type=int, default=DEFAULT_MAX_STALLED,
                    help="more stalled rows than this at once -> cancel none")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only: no DB writes, no scancel")
    args = ap.parse_args()
    if not args.db:
        print("[cron_release] no DB path (set --db or $SJM_DB)", file=sys.stderr)
        return 2

    db = JobDB(args.db)
    report = db.cron_release(owner_state, cancel, stall_s=args.stall_s,
                             max_stalled=args.max_stalled, dry_run=args.dry_run,
                             partition_of=partition_of)
    tag = "[cron_release]" + (" DRY-RUN" if args.dry_run else "")
    for name, job_id, part in report.handed_off:
        print(f"{tag} handoff {name} (owner {job_id} on {part}: trained, in-process "
              f"final_eval) -> scancelled, eval lane takes over")
    try:
        lane = eval_lane.top_up(db, dry_run=args.dry_run)
    except Exception as exc:  # never let the top-up hide the release report
        lane = {"promoted": [], "unowned": "?", "pending_jobs": None, "submitted": []}
        print(f"{tag} eval top-up FAILED: {exc}")
    for name in lane["promoted"]:
        print(f"{tag} promoted {name} (pending, fully trained) -> eval")
    if lane["submitted"]:
        print(f"{tag} eval lane: {lane['unowned']} unowned eval row(s), "
              f"{lane['pending_jobs']} queued sjm-eval job(s) -> submitted {lane['submitted']}")
    for name, job_id, reason in report.released:
        print(f"{tag} released {name} (owner {job_id}: {reason})")
    cancelled = {(name, job_id) for name, job_id, _ in report.cancelled}
    for name, job_id, age in report.stalled:
        outcome = ("scancelled" if (name, job_id) in cancelled
                   else "NOT cancelled (breaker)" if report.breaker_tripped
                   else "scancel FAILED")
        print(f"{tag} stalled {name} (owner {job_id}, heartbeat {_age(age)} old): {outcome}")
    for name, job_id in report.unknown:
        print(f"{tag} probe failed for {name} (owner {job_id}); left alone")
    if report.breaker_tripped:
        print(f"{tag} BREAKER: {len(report.stalled)} rows stalled at once "
              f"(> {args.max_stalled}); cancelled none")
    print(f"{tag} summary released={len(report.released)} stalled={len(report.stalled)} "
          f"cancelled={len(report.cancelled)} breaker={int(report.breaker_tripped)} "
          f"unknown={len(report.unknown)} handoff={len(report.handed_off)} "
          f"eval_unowned={lane['unowned']} eval_submitted={len(lane['submitted'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
