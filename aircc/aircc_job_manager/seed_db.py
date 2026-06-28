#!/usr/bin/env python3
"""Seed the AIRCC monitoring DB from a CSV -- selectively by arch + init.

Arch and seed (init) are chosen here, at launch time. The full CSVs always
contain every model; seeding decides which subset actually enters the live queue.
Idempotent: re-running with another ``--init`` extends the queue without
disturbing already-running / finished rows.

    python -m aircc.aircc_job_manager.seed_db --db /path/aircc_jobs.sqlite \
        --arch convnext_small --init 1 --init 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aircc.aircc_job_manager.csv_spec import ARCHES, CSV_DIR, load_arch_rows
from aircc.aircc_job_manager.db import AirccDB


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed the AIRCC DB from a CSV (selective by arch+init).")
    ap.add_argument("--db", required=True, help="SQLite DB path (created if absent).")
    ap.add_argument("--arch", required=True, choices=ARCHES)
    ap.add_argument("--init", type=int, action="append", default=None,
                    help="Init group(s) to seed (repeatable). Default: 0,1,2.")
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--reconcile", action="store_true",
                    help="Re-open finished rows whose CSV total_epochs now exceeds their progress.")
    args = ap.parse_args()

    inits = set(args.init) if args.init else {0, 1, 2}
    rows = [r for r in load_arch_rows(args.arch, args.csv_dir) if int(r["init"]) in inits]
    db = AirccDB(args.db)

    if args.reconcile:
        reopened = sum(db.reconcile(r["model_name"], int(r["training.epochs"])) for r in rows)
        print(f"[seed_db] reconcile {args.arch} (inits={sorted(inits)}): "
              f"re-opened {reopened} finished model(s) for more epochs -> {args.db}")
        return

    for r in rows:
        db.upsert_pending(r["model_name"], int(r["training.epochs"]), int(r["priority"]))
    print(f"[seed_db] seeded {len(rows)} models from {args.arch} "
          f"(inits={sorted(inits)}) -> {args.db}")


if __name__ == "__main__":
    main()
