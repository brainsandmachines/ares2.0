"""In-job DB control used by ``sbatches/stage_runner.sh`` (runs on the cluster).

Reads ``ORCH_DB`` and ``ORCH_MODEL_ID`` from the environment (exported by the
orchestrator via ``sbatch --export``). stdlib + ``orchestrator.db`` only, so it
imports cleanly in the training conda env.

Subcommands:
    get <field>            print one row field (status|current_stage|
                           total_epochs|current_epoch|model_dir|launcher|job_name)
    advance <stage>        set current_stage (COMPLETED also sets status)
    set-status <status>    set status
    tracked                exit 0 if ORCH_DB+ORCH_MODEL_ID are set, else 1
"""

from __future__ import annotations

import argparse
import os
import sys

from .db import OrchestratorDB


def _env() -> tuple[str, str]:
    db_path = os.environ.get("ORCH_DB")
    model_id = os.environ.get("ORCH_MODEL_ID")
    if not db_path or not model_id:
        print("not tracked (ORCH_DB/ORCH_MODEL_ID unset)", file=sys.stderr)
        sys.exit(2)
    return db_path, model_id


def main() -> None:
    ap = argparse.ArgumentParser(prog="jobctl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get"); g.add_argument("field")
    a = sub.add_parser("advance"); a.add_argument("stage")
    s = sub.add_parser("set-status"); s.add_argument("status")
    sub.add_parser("tracked")
    args = ap.parse_args()

    if args.cmd == "tracked":
        ok = bool(os.environ.get("ORCH_DB") and os.environ.get("ORCH_MODEL_ID"))
        sys.exit(0 if ok else 1)

    db_path, model_id = _env()
    db = OrchestratorDB(db_path)

    if args.cmd == "get":
        row = db.get_model_state(model_id)
        if row is None:
            print("", end="")
            return
        field = args.field
        if field == "total_epochs":
            print(row.total_epochs)
        else:
            print(getattr(row, field))
    elif args.cmd == "advance":
        db.advance_stage(model_id, args.stage)
    elif args.cmd == "set-status":
        db.set_status(model_id, args.status)


if __name__ == "__main__":
    main()
