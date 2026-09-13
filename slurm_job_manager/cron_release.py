#!/usr/bin/env python3
"""Hourly dead/stalled-owner sweep over the sjm DB (``JobDB.cron_release``).

Runs on the cluster login node, driven by ``scripts/cron_release.sh`` from a Botero cron::

    python -m slurm_job_manager.cron_release --db $SJM_DB [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional

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

    report = JobDB(args.db).cron_release(owner_state, cancel, stall_s=args.stall_s,
                                         max_stalled=args.max_stalled, dry_run=args.dry_run)
    tag = "[cron_release]" + (" DRY-RUN" if args.dry_run else "")
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
          f"unknown={len(report.unknown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
