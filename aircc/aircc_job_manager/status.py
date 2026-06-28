#!/usr/bin/env python3
"""Read-only AIRCC status dashboard.

    python -m aircc.aircc_job_manager.status --db /path/aircc_jobs.sqlite

Shows counts by status, currently-running models (epoch cur/total + heartbeat
age), stale running models (>15 min), recently finished (with best_score),
failed (with last_error), and dependency-blocked pending models. Never writes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aircc.aircc_job_manager.csv_spec import CSV_DIR, deps_map, load_all_rows
from aircc.aircc_job_manager.db import STALE_SECONDS, AirccDB


def _age(ts: int | None, now: int) -> str:
    if not ts:
        return "n/a"
    s = now - int(ts)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60}m"


def main() -> None:
    ap = argparse.ArgumentParser(description="AIRCC status dashboard (read-only).")
    ap.add_argument("--db", required=True)
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--recent", type=int, default=10, help="How many recent finished to show.")
    args = ap.parse_args()

    now = int(time.time())
    db = AirccDB(args.db)
    jobs = db.all_jobs()
    by_name = {j.model_name: j for j in jobs}
    deps = deps_map(load_all_rows(args.csv_dir))

    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1

    print("=== AIRCC status ===")
    print("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"  (total {len(jobs)})")

    running = [j for j in jobs if j.status == "running"]
    stale = [j for j in running if not j.heartbeat_ts or now - j.heartbeat_ts > STALE_SECONDS]
    print(f"\n-- running ({len(running)}) --")
    for j in sorted(running, key=lambda x: x.model_name):
        flag = "  STALE" if j in stale else ""
        print(f"  {j.model_name:55s} ep {j.current_epoch}/{j.total_epochs}  "
              f"hb {_age(j.heartbeat_ts, now)}  task {j.owner_task}{flag}")

    if stale:
        print(f"\n-- stale running ({len(stale)}, >15m no heartbeat; will be requeued) --")
        for j in stale:
            print(f"  {j.model_name}  hb {_age(j.heartbeat_ts, now)}")

    finished = [j for j in jobs if j.status == "finished" and j.heartbeat_ts]
    finished.sort(key=lambda x: x.heartbeat_ts or 0, reverse=True)
    print(f"\n-- recently finished (top {args.recent}) --")
    for j in finished[: args.recent]:
        score = f"{j.best_score:.1f}%" if j.best_score is not None else "n/a"
        print(f"  {j.model_name:55s} best={score}  {_age(j.heartbeat_ts, now)} ago")

    failed = [j for j in jobs if j.status == "failed"]
    print(f"\n-- failed ({len(failed)}) --")
    for j in sorted(failed, key=lambda x: x.model_name):
        print(f"  {j.model_name}: {(j.last_error or '').splitlines()[0] if j.last_error else ''}")

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
