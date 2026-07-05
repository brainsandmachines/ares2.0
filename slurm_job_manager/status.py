#!/usr/bin/env python3
"""Read-only status dashboard for the Slurm job manager.

    python -m slurm_job_manager.status --db $SJM_DB

Counts by status, currently-running models (epoch + heartbeat age + owner),
recently finished (best_score), failed (signature + first error line), the test
lane, and dependency-blocked pending models. Never writes.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .csv_spec import CSV_DIR, deps_map, load_all_rows
from .db import JobDB


def _age(ts, now: int) -> str:
    if not ts:
        return "n/a"
    s = now - int(ts)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60}m"


def main() -> None:
    ap = argparse.ArgumentParser(description="Job manager status (read-only).")
    ap.add_argument("--db", default=os.environ.get("SJM_DB"))
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--recent", type=int, default=10)
    args = ap.parse_args()
    if not args.db:
        ap.error("no DB path (set --db or $SJM_DB)")

    now = int(time.time())
    db = JobDB(args.db)
    jobs = db.all_jobs()
    by_name = {j.model_name: j for j in jobs}
    deps = deps_map(load_all_rows(args.csv_dir))

    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1

    print("=== job manager status ===")
    print("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) +
          f"  (total {len(jobs)})")

    running = [j for j in jobs if j.status == "running"]
    print(f"\n-- running ({len(running)}) --")
    for j in sorted(running, key=lambda x: (-x.is_test, x.priority)):
        tag = " [TEST]" if j.is_test else ""
        print(f"  {j.model_name:55s} ep {j.current_epoch}/{j.total_epochs}  "
              f"hb {_age(j.heartbeat_ts, now)}  job {j.slurm_job_id}"
              f"  requeued {j.requeued}{tag}")

    pending_test = [j for j in jobs if j.status == "pending" and j.is_test]
    if pending_test:
        print(f"\n-- test lane pending ({len(pending_test)}, claimed first) --")
        for j in sorted(pending_test, key=lambda x: x.priority):
            print(f"  {j.model_name:55s} priority {j.priority}")

    finished = [j for j in jobs if j.status == "finished" and j.heartbeat_ts]
    finished.sort(key=lambda x: x.heartbeat_ts or 0, reverse=True)
    print(f"\n-- recently finished (top {args.recent}) --")
    for j in finished[: args.recent]:
        score = f"{j.best_score:.1f}%" if j.best_score is not None else "n/a"
        print(f"  {j.model_name:55s} best={score}  {_age(j.heartbeat_ts, now)} ago")

    failed = [j for j in jobs if j.status == "failed"]
    print(f"\n-- failed ({len(failed)}) --")
    for j in sorted(failed, key=lambda x: x.model_name):
        first = (j.last_error or "").strip().splitlines()
        sig = f"[{j.last_error_hash[:8]}] " if j.last_error_hash else ""
        print(f"  {j.model_name:55s} {sig}{first[-1] if first else ''}")

    blocked = []
    for j in jobs:
        if j.status != "pending":
            continue
        dep = deps.get(j.model_name, "")
        if not dep:
            continue
        dj = by_name.get(dep)
        if not (dj and dj.status == "finished" and (dj.best_checkpoint or "").strip()):
            blocked.append((j, dep))
    print(f"\n-- dependency-blocked pending ({len(blocked)}) --")
    for j, dep in sorted(blocked, key=lambda t: t[0].priority)[:40]:
        dstat = by_name[dep].status if dep in by_name else "absent"
        print(f"  {j.model_name:55s} waits on {dep} ({dstat})")
    if len(blocked) > 40:
        print(f"  ... and {len(blocked) - 40} more")


if __name__ == "__main__":
    main()
