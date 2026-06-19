"""Operator CLI for the orchestrator.

Examples:
    python -m orchestrator.cli status
    python -m orchestrator.cli validate-model l2_8_init1
    python -m orchestrator.cli force-stage l2_8_init1 AA_SWEEP --kinds last,advbest
    python -m orchestrator.cli tick --dry-run
    python -m orchestrator.cli requeue-stale
    python -m orchestrator.cli seed
"""

from __future__ import annotations

import argparse

from .config import Config
from .core import Orchestrator, SlurmClient
from .db import (
    OrchestratorDB,
    STAGE_AA_SWEEP,
    STAGE_COMPLETED,
    STAGE_PLOT_RESULTS,
    STAGE_TRAIN,
    STAGE_CUSTOM_TASK,
)
from . import seed as seed_mod

VALID_STAGES = {
    STAGE_TRAIN, STAGE_AA_SWEEP, STAGE_PLOT_RESULTS, STAGE_CUSTOM_TASK, STAGE_COMPLETED
}


def _db(cfg: Config) -> OrchestratorDB:
    return OrchestratorDB(cfg.db_path)


def cmd_status(cfg: Config, args) -> None:
    db = _db(cfg)
    rows = db.list_models(status=args.filter)
    hdr = (f"{'model_id':28} {'arch':6} {'proto':8} {'eps':>4} {'status':9} "
           f"{'stage':13} {'epoch':>9} {'job':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ep = f"{r.current_epoch}/{r.total_epochs}"
        arch = (r.arch or "").replace("convnext_", "")
        eps = "-" if r.eps is None else f"{r.eps:g}"
        print(f"{r.model_id:28.28} {arch:6.6} {(r.protocol or '-'):8.8} {eps:>4} "
              f"{r.status:9} {r.current_stage:13} {ep:>9} {str(r.slurm_job_id or '-'):>8}")
    print(f"\n{len(rows)} rows")


def cmd_validate(cfg: Config, args) -> None:
    db = _db(cfg)
    row = db.get_model_state(args.model_id)
    if not row:
        print(f"no such model: {args.model_id}")
        return
    print(f"model_id : {row.model_id}")
    print(f"arch     : {row.arch}")
    print(f"protocol : {row.protocol}")
    print(f"eps      : {row.eps}")
    print(f"status   : {row.status}")
    print(f"stage    : {row.current_stage}")
    print(f"epoch    : {row.current_epoch}/{row.total_epochs}")
    print(f"node     : {row.cluster_node}")
    print(f"slurm_job: {row.slurm_job_id}")
    print(f"model_dir: {row.model_dir}")
    if row.slurm_job_id:
        slurm = SlurmClient(cfg)
        try:
            state = slurm.job_state(row.slurm_job_id)
            active = slurm.is_active(row.slurm_job_id)
            print(f"squeue   : state={state} active={active}")
            if row.status == "RUNNING" and not active:
                print("  WARNING: row is RUNNING but job is not active (zombie).")
        except Exception as exc:
            print(f"squeue   : (ssh failed: {exc})")


def cmd_force_stage(cfg: Config, args) -> None:
    db = _db(cfg)
    if args.stage not in VALID_STAGES:
        raise SystemExit(f"invalid stage {args.stage}; one of {sorted(VALID_STAGES)}")
    db.force_stage(args.model_id, args.stage, status=args.status)
    print(f"forced {args.model_id} -> stage={args.stage} status={args.status}"
          + (f" kinds={args.kinds}" if args.kinds else ""))
    if args.kinds:
        # Recorded for the operator's intent; the AA stage reads kinds from its
        # own override env when the job runs. Surface it clearly here.
        print("note: pass AA_KINDS to the launcher to limit checkpoint kinds.")


def cmd_tick(cfg: Config, args) -> None:
    db = _db(cfg)
    orch = Orchestrator(db, SlurmClient(cfg), cfg)
    subs = orch.tick(dry_run=args.dry_run)
    tag = "[dry-run] would submit" if args.dry_run else "submitted"
    for s in subs:
        print(f"{tag}: {s.model_id} -> {s.node} ({s.launcher}) job={s.slurm_job_id}")
    if not subs:
        print("no free slots / no eligible PENDING work")


def cmd_requeue_stale(cfg: Config, args) -> None:
    db = _db(cfg)
    n = db.requeue_stale_running(int(cfg.stale_running_hours * 3600))
    print(f"requeued {n} stale RUNNING rows")


def cmd_seed(cfg: Config, args) -> None:
    n = seed_mod.seed(cfg.db_path, args.experiments)
    print(f"seeded {n} experiments into {cfg.db_path}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="orchestrator")
    ap.add_argument("--env", default=None, help="path to .env (default orchestrator/.env)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status"); p.add_argument("--filter", default=None); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("validate-model"); p.add_argument("model_id"); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("force-stage")
    p.add_argument("model_id"); p.add_argument("stage")
    p.add_argument("--status", default="PENDING"); p.add_argument("--kinds", default=None)
    p.set_defaults(fn=cmd_force_stage)
    p = sub.add_parser("tick"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_tick)
    p = sub.add_parser("requeue-stale"); p.set_defaults(fn=cmd_requeue_stale)
    import os as _os
    _default_exp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "experiments.yaml")
    p = sub.add_parser("seed"); p.add_argument("--experiments", default=_default_exp); p.set_defaults(fn=cmd_seed)

    args = ap.parse_args()
    cfg = Config.load(args.env)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
