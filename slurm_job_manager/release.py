#!/usr/bin/env python3
"""Release a claimed row back to the pending pool (the SIGTERM-trap entrypoint).

The manager sbatch traps Slurm's time-limit SIGTERM and runs::

    python -m slurm_job_manager.release "$(cat $SJM_TRAP_FILE)"

which returns the in-flight model to ``pending`` (clears the Slurm owner, bumps
``requeued``) so the next array task resumes it from ``last.pth.tar``. Safe to
call with an empty/absent model name (no-op) and safe if the row already
finished/failed (the release only touches a running/pending row).
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
    args = ap.parse_args()

    name = (args.model_name or "").strip()
    if not name:
        return 0
    if not args.db:
        print("[release] no DB path (set --db or $SJM_DB)", file=sys.stderr)
        return 2
    n = JobDB(args.db).release(name)
    print(f"[release] {name}: {'released' if n else 'no-op (already terminal)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
