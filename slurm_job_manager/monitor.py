"""Botero-side failure escalation pass (run on demand or from an hourly cron).

The cluster controller already marks real failures FAILED and stores the log tail
+ a deterministic signature on the row. This pass is the **codex escalation**:
for each FAILED row carrying a stored log, run the deterministic failure analyzer
once per never-before-seen signature -- writing a markdown report under
``recommendations/`` and (optionally) emailing it. Repeat signatures are skipped
via the ``failure_hashes`` dedup store, so this is safe to run every hour.

It does NOT top up pools or manage capacity -- GPU scaling is manual (resize the
per-partition array). Pure DB reads + the analyzer; no ssh/sacct needed.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from .config import Config
from .db import JobDB
from .failure_analyzer import analyze_failure
from . import notify

logger = logging.getLogger("sjm.monitor")


def run_once(cfg: Optional[Config] = None, dry_run: bool = False) -> int:
    cfg = cfg or Config.load()
    db = JobDB(cfg.db_path)
    llm = notify.make_llm_client() if cfg.enable_failure_analyzer else None
    emailer = notify.make_emailer(cfg)

    failed = [j for j in db.all_jobs() if j.status == "failed" and (j.last_error or "").strip()]
    escalated = 0
    for job in failed:
        # analyze_failure recomputes the signature from the stored log and dedups
        # via failure_hashes, so re-processing an already-seen row is a cheap no-op.
        raw = job.last_error or ""
        if dry_run:
            logger.info("[dry-run] would analyze %s", job.model_name)
            continue
        outcome = analyze_failure(db, job.model_name, raw, cfg.recommendations_dir,
                                  llm_client=llm, emailer=emailer)
        if outcome.is_new:
            escalated += 1
    logger.info("monitor pass done%s -- %d failed row(s), %d new signature(s) escalated",
                " (dry-run)" if dry_run else "", len(failed), escalated)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="slurm_job_manager.monitor")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
