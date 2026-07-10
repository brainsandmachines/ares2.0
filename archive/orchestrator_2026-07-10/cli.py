"""Operator CLI for the orchestrator.

Examples:
    python -m orchestrator.cli status
    python -m orchestrator.cli status --filter TRAINING
    python -m orchestrator.cli validate-model l2_8_init1
    python -m orchestrator.cli force-status l2_8_init1 AA_EVAL
    python -m orchestrator.cli requeue l2_8_init1
    python -m orchestrator.cli monitor
    python -m orchestrator.cli seed
"""

from __future__ import annotations

import argparse

from .config import Config
from .core import SlurmClient
from .db import OrchestratorDB, STATUSES, ACTIVE_STATUSES
from . import seed as seed_mod


def _db(cfg: Config) -> OrchestratorDB:
    return OrchestratorDB(cfg.db_path)


def cmd_status(cfg: Config, args) -> None:
    db = _db(cfg)
    rows = db.list_models(status=args.filter)
    if getattr(args, "active", False):
        # Only rows a controller task currently owns = actually running now.
        rows = [r for r in rows if r.slurm_job_id is not None
                and r.status in ACTIVE_STATUSES]
    hdr = (f"{'model_id':28} {'arch':6} {'proto':8} {'eps':>4} {'status':9} "
           f"{'epoch':>9} {'rq':>3} {'best':>8} {'job':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ep = f"{r.current_epoch}/{r.target_epoch}"
        arch = (r.arch or "").replace("convnext_", "")
        eps = "-" if r.eps is None else f"{r.eps:g}"
        print(f"{r.model_id:28.28} {arch:6.6} {(r.protocol or '-'):8.8} {eps:>4} "
              f"{r.status:9} {ep:>9} {r.requeued:>3} "
              f"{(r.best_checkpoint or '-'):>8.8} {str(r.slurm_job_id or '-'):>10}")
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
    print(f"epoch    : {row.current_epoch}/{row.target_epoch} (next={row.next_epoch})")
    print(f"requeued : {row.requeued}")
    print(f"best_ckpt: {row.best_checkpoint}"
          + ("" if row.best_score is None else f" (score={row.best_score:g})"))
    print(f"node     : {row.cluster_node}")
    print(f"slurm_job: {row.slurm_job_id}")
    print(f"model_dir: {row.model_dir}")
    if row.slurm_job_id:
        slurm = SlurmClient(cfg)
        try:
            state = slurm.job_state(row.slurm_job_id)
            active = slurm.is_active(row.slurm_job_id)
            print(f"squeue   : state={state} active={active}")
            if row.status in ACTIVE_STATUSES and not active:
                print("  WARNING: row claims an owner but the job is not active (zombie).")
        except Exception as exc:
            print(f"squeue   : (ssh failed: {exc})")


def cmd_force_status(cfg: Config, args) -> None:
    db = _db(cfg)
    if args.status not in STATUSES:
        raise SystemExit(f"invalid status {args.status}; one of {sorted(STATUSES)}")
    db.force_status(args.model_id, args.status)
    print(f"forced {args.model_id} -> status={args.status} (ownership released)")


def cmd_requeue(cfg: Config, args) -> None:
    db = _db(cfg)
    if db.get_model_state(args.model_id) is None:
        raise SystemExit(f"no such model: {args.model_id}")
    db.requeue(args.model_id)
    print(f"requeued {args.model_id} (slurm owner cleared; now claimable)")


def cmd_monitor(cfg: Config, args) -> None:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    from .monitor import run_once
    run_once(cfg, dry_run=args.dry_run)


def cmd_seed(cfg: Config, args) -> None:
    n = seed_mod.seed(cfg.db_path, args.experiments)
    print(f"seeded {n} experiments into {cfg.db_path}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="orchestrator")
    ap.add_argument("--env", default=None, help="path to .env (default orchestrator/.env)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p.add_argument("--filter", default=None)
    p.add_argument("--active", action="store_true",
                   help="only models a controller is running right now")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("validate-model"); p.add_argument("model_id"); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("force-status")
    p.add_argument("model_id"); p.add_argument("status")
    p.set_defaults(fn=cmd_force_status)
    p = sub.add_parser("requeue"); p.add_argument("model_id"); p.set_defaults(fn=cmd_requeue)
    p = sub.add_parser("monitor"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_monitor)
    import os as _os
    _default_exp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "experiments.yaml")
    p = sub.add_parser("seed"); p.add_argument("--experiments", default=_default_exp); p.set_defaults(fn=cmd_seed)

    args = ap.parse_args()
    cfg = Config.load(args.env)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
