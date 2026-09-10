#!/usr/bin/env python3
"""Release a claimed row back to the pending pool (the SIGTERM-trap entrypoint).

The manager sbatch traps Slurm's time-limit SIGTERM and runs::

    python -m slurm_job_manager.release "$(cat $SJM_TRAP_FILE)"

which returns the in-flight model to ``pending`` (clears the Slurm owner, bumps
``requeued``) so the next array task resumes it from ``last.pth.tar``. Safe to
call with an empty/absent model name (no-op) and safe if the row already
finished/failed (the release only touches a running/pending row).

The release is guarded by ``$SLURM_JOB_ID``: a task whose claim was taken over
by another task no-ops instead of freeing the row out from under the run that
now owns it. Manual use from a login shell has no ``SLURM_JOB_ID`` and so is
unguarded; inside an salloc, pass ``--any-owner`` to get that behaviour.
"""

from __future__ import annotations

import argparse
import os
import sys

from .db import JobDB


def main() -> int:
    ap = argparse.ArgumentParser(prog="slurm_job_manager.release")
    ap.add_argument("model_name", nargs="?", default="")
    ap.add_argument("--db", default=os.environ.get("SJM_DB"))
    ap.add_argument("--owner", default=os.environ.get("SLURM_JOB_ID"),
                    help="release only if this Slurm job still owns the row "
                         "(defaults to $SLURM_JOB_ID, i.e. the trapping task)")
    ap.add_argument("--any-owner", action="store_true",
                    help="manual override: release whoever owns the row")
    args = ap.parse_args()

    name = (args.model_name or "").strip()
    if not name:
        return 0
    if not args.db:
        print("[release] no DB path (set --db or $SJM_DB)", file=sys.stderr)
        return 2
    owner = None if args.any_owner else (int(args.owner) if args.owner else None)
    n = JobDB(args.db).release(name, owner=owner)
    print(f"[release] {name}: "
          f"{'released' if n else 'no-op (not owner / already terminal)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
