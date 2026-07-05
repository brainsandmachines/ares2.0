#!/usr/bin/env python3
"""Seed / reconcile the job DB from the CSVs -- the "add a job" flow.

Two-step workflow to add or change a job:
  1. edit ``slurm_job_manager/csv/<arch>.csv`` (add a row / raise its epochs), then
  2. run this to make the DB aware of it.

Idempotent: re-seeding never disturbs already-running / finished rows -- it only
refreshes total_epochs / priority / is_test. ``--reconcile`` additionally re-opens
finished rows whose CSV ``training.epochs`` now exceeds their recorded progress
(so training resumes the delta from its own last.pth.tar).

    python -m slurm_job_manager.seed --db $SJM_DB --arch convnext_small --init 1 --init 2
    python -m slurm_job_manager.seed --db $SJM_DB --arch convnext_small --reconcile
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .csv_spec import ARCHES, CSV_DIR, is_test_row, load_arch_rows
from .db import JobDB


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed the job DB from a CSV (selective by arch+init).")
    ap.add_argument("--db", default=os.environ.get("SJM_DB"), help="SQLite DB path (default $SJM_DB).")
    ap.add_argument("--arch", required=True, choices=ARCHES)
    ap.add_argument("--init", type=int, action="append", default=None,
                    help="Init group(s) to seed (repeatable). Default: all present in the CSV.")
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--reconcile", action="store_true",
                    help="Re-open finished rows whose CSV total_epochs now exceeds their progress.")
    args = ap.parse_args()
    if not args.db:
        ap.error("no DB path (set --db or $SJM_DB)")

    rows = load_arch_rows(args.arch, args.csv_dir)
    if args.init is not None:
        inits = set(args.init)
        rows = [r for r in rows if int(r["init"]) in inits]
    db = JobDB(args.db)

    if args.reconcile:
        reopened = sum(db.reconcile(r["model_name"], int(r["training.epochs"])) for r in rows)
        print(f"[seed] reconcile {args.arch}: re-opened {reopened} finished model(s) "
              f"for more epochs -> {args.db}")
        return

    for r in rows:
        db.upsert_pending(r["model_name"], int(r["training.epochs"]),
                          int(r["priority"]), is_test=is_test_row(r))
    n_test = sum(1 for r in rows if is_test_row(r))
    print(f"[seed] seeded {len(rows)} models from {args.arch} ({n_test} test) -> {args.db}")


if __name__ == "__main__":
    main()
