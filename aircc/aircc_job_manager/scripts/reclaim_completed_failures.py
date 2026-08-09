#!/usr/bin/env python3
"""Reclaim `failed` rows whose training actually completed on disk.

lifecycle.run()'s own final DB write (best-checkpoint + mark_finished) can hit the
same transient sqlite CANTOPEN on the shared FS as the in-training hook does (see
backfill_best_checkpoints.py). But when it happens on that *last* write, the
exception escapes lifecycle.run() entirely and job_manager's outer handler calls
mark_failed -- so a run that fully trained and evaluated ends up recorded `failed`,
not `finished` with a missing best_checkpoint. backfill_best_checkpoints.py only
scans `status='finished'` rows, so it never sees these.

This script targets `status='failed'` rows whose `last_error` matches one of
``db.TRANSIENT_ERROR_MARKERS`` -- a literal DB-write-failure signature, never a
training error -- AND that resolve a real checkpoint via `best_checkpoint_for_threat`,
i.e. there is actually scorable AutoAttack output on disk. Only those get flipped to
`finished`. A row matching the error text but with no resolvable output is left
`failed` and reported as unresolved: the error text alone doesn't prove the run
completed, and it still needs a normal failure diagnosis.

Safe to re-run -- idempotent, only touches rows in the state described above.

Run on the AIRCC login node (the DB is not writable through the read-only mount):

    python -m aircc.aircc_job_manager.scripts.reclaim_completed_failures           # dry run
    python -m aircc.aircc_job_manager.scripts.reclaim_completed_failures --apply
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
from aircc.aircc_job_manager.db import TRANSIENT_ERROR_MARKERS, AirccDB  # noqa: E402


def _is_transient_db_error(last_error: str | None) -> bool:
    text = last_error or ""
    return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)


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
               if j.status == "failed" and _is_transient_db_error(j.last_error)]
    if not targets:
        print("nothing to reclaim: no failed row has a transient DB-write error")
        print("RECLAIMED:")
        return 0

    print(f"{'MODEL':<58} {'SCORE':>10}  CHECKPOINT")
    reclaimed: list[str] = []
    unresolved = 0
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
            print(f"{job.model_name:<58} {'--':>10}  no scorable AutoAttack output in "
                  f"{model_dir} (left failed -- needs a real diagnosis)")
            unresolved += 1
            continue
        print(f"{job.model_name:<58} {score:>10.4f}  {Path(path).name}")
        if args.apply:
            if not (job.best_checkpoint or "").strip():
                db.set_best_checkpoint(job.model_name, path, score)
            db.mark_finished(job.model_name)
            reclaimed.append(job.model_name)

    if args.apply:
        print(f"\nreclaimed {len(reclaimed)} row(s); {unresolved} unresolved (left failed)")
    else:
        print(f"\nDRY RUN -- {len(targets) - unresolved} row(s) would be reclaimed, "
              f"{unresolved} unresolved. Re-run with --apply.")
    # Machine-parseable for daily_monitor.repair_falsely_failed_jobs -- only ever
    # non-empty on an --apply run, only lists rows actually flipped to finished.
    print(f"RECLAIMED:{','.join(reclaimed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
