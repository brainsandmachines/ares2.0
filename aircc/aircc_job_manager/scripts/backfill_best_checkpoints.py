#!/usr/bin/env python3
"""Backfill best_checkpoint/best_score for finished rows the in-training hook missed.

``progress.write_best_checkpoint`` gets exactly one shot per run and swallows DB
errors, so a transient sqlite CANTOPEN on the shared FS leaves a row ``finished``
with a NULL best_checkpoint even though the AutoAttack eval completed and its
scores are on disk. Those rows then silently gate every continuation row that
depends on them (``claim_next`` requires dep finished AND best_checkpoint set).

``lifecycle.ensure_best_checkpoint`` now closes that hole for new runs; this
script repairs rows that already finished without one. Safe to re-run -- it only
touches rows where best_checkpoint IS NULL, and re-derives the score from the
same on-disk AutoAttack output the hook would have used.

Run on the AIRCC login node (the DB is not writable through the read-only mount):

    python -m aircc.aircc_job_manager.scripts.backfill_best_checkpoints           # dry run
    python -m aircc.aircc_job_manager.scripts.backfill_best_checkpoints --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aircc.aircc_job_manager.best_checkpoint import best_checkpoint_for_threat  # noqa: E402
from aircc.aircc_job_manager.csv_spec import CSV_DIR, load_spec  # noqa: E402
from aircc.aircc_job_manager.db import AirccDB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get(
        "AIRCC_DB", str(REPO_ROOT / "aircc" / "aircc_job_manager" / "aircc_jobs.sqlite")))
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--models-root", type=Path, default=Path(
        os.environ.get("MODELS_ROOT", REPO_ROOT / "results" / "models")))
    ap.add_argument("--apply", action="store_true",
                    help="write to the DB (default: dry run, report only)")
    args = ap.parse_args()

    db = AirccDB(args.db)
    _, rows, _ = load_spec(args.csv_dir)

    targets = [j for j in db.all_jobs()
               if j.status == "finished" and not (j.best_checkpoint or "").strip()]
    if not targets:
        print("nothing to backfill: every finished row has a best_checkpoint")
        return 0

    print(f"{'MODEL':<58} {'SCORE':>10}  CHECKPOINT")
    fixed = failed = 0
    for job in targets:
        row = rows.get(job.model_name, {})
        norm = str(row.get("threat_norm", "") or "").strip() or None
        eps_raw = str(row.get("threat_eps", "") or "").strip()
        eps = float(eps_raw) if eps_raw else None
        model_dir = args.models_root / job.model_name
        try:
            path, score = best_checkpoint_for_threat(model_dir, norm, eps)
        except Exception as exc:
            path, score = None, None
            print(f"{job.model_name:<58} {'ERROR':>10}  {exc}")
        if not path:
            if score is None and path is None:
                print(f"{job.model_name:<58} {'--':>10}  no scorable AutoAttack output in {model_dir}")
            failed += 1
            continue
        print(f"{job.model_name:<58} {score:>10.4f}  {Path(path).name}")
        if args.apply:
            db.set_best_checkpoint(job.model_name, path, score)
            fixed += 1

    if args.apply:
        print(f"\nwrote {fixed} row(s); {failed} unresolvable")
    else:
        print(f"\nDRY RUN -- {len(targets) - failed} row(s) would be written, "
              f"{failed} unresolvable. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
