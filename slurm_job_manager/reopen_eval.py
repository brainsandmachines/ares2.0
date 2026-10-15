#!/usr/bin/env python3
"""Re-open finished models whose AutoAttack grid is incomplete (eval-only rerun).

The training job runs the full linf/l2/l1 x eps AutoAttack grid in-process
(``final_eval_aatype: full_sweep``). A model can still finish without it: it was
already running when that default changed (old config loaded at start), or its
final_eval raised -- the exception is logged, the process exits 0, and the row
is marked finished anyway. This finds those rows and, with ``--apply``, puts them
back to pending. The re-claim resumes from its own ``last.pth.tar`` at the final
epoch, trains nothing, and the in-process final_eval fills only the missing cells.

    python -m slurm_job_manager.reopen_eval --db $SJM_DB               # dry run: list
    python -m slurm_job_manager.reopen_eval --db $SJM_DB --apply --all
    python -m slurm_job_manager.reopen_eval --db $SJM_DB --apply --model swin_b/l2trades_4_init2

The grid checked is the one final_eval would run: ``final_eval_aa_eps_inputs`` and
``final_eval_aa_checkpoint_kinds`` from ``robust_training/configs/config.yaml``,
on the three norms, with the same completeness rule (``is_complete_output``).
Models in ``aa_sweep/sweep_excludes.csv`` (discarded runs: catastrophic overfitting,
gradient masking) are never re-opened, matching the sweep's own census.

Also skipped: models the aa_sweep cluster queue still has pending/running units
for (both would write the same CSVs). ``--all`` re-opens a model at most
``--max-reopens`` times (history in ``reopen_eval_history.json`` next to the DB),
so an eval that keeps failing is reported as ``capped`` instead of relaunched
daily; ``--model NAME`` ignores the cap. The last line is a parseable summary for
the daily cron (``scripts/reopen_eval_daily.sh``).

Caveat: while a re-opened row is pending/running, its not-yet-claimed dependents
are blocked again (the claim gate wants the dependency finished).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from .db import JobDB, STATUS_FINISHED
from .lifecycle import REPO_ROOT

CONFIG_YAML = REPO_ROOT / "robust_training" / "configs" / "config.yaml"


def configured_grid(config_yaml: Path = CONFIG_YAML):
    """(norms, eps_inputs, checkpoint_kinds) final_eval runs with the config defaults."""
    from omegaconf import OmegaConf

    from ares.utils.final_eval_helpers import (
        AA_FULL_SWEEP_NORMS,
        _parse_checkpoint_kinds,
        _parse_float_list,
    )

    cfg = OmegaConf.load(config_yaml)
    eps_inputs = _parse_float_list(cfg.get("final_eval_aa_eps_inputs"))
    kinds = _parse_checkpoint_kinds(cfg.get("final_eval_aa_checkpoint_kinds"))
    return list(AA_FULL_SWEEP_NORMS), eps_inputs, kinds


def missing_cells(model_dir: Path, norms, eps_inputs, kinds) -> dict[str, int]:
    """{kind: missing cell count} for each kind whose checkpoint exists and is incomplete."""
    from data_analysis.autoattack_array_eval import (
        _setting_key,
        find_checkpoint_for_kind,
        observed_settings,
        output_csv_for_checkpoint_kind,
    )

    expected = {_setting_key(n, e) for n in norms for e in eps_inputs}
    out: dict[str, int] = {}
    for kind in kinds:
        ckpt = find_checkpoint_for_kind(model_dir, kind)
        if ckpt is None:
            continue
        csv_path = model_dir / output_csv_for_checkpoint_kind("autoattack_sweep_results.csv", kind)
        found = observed_settings(csv_path, checkpoint_path=ckpt, model_name=model_dir.name)
        n_missing = len(expected - found)
        if n_missing:
            out[kind] = n_missing
    return out


def aa_queue_active_models(queue_db: Path) -> set[str]:
    """Models with a pending/running unit in the aa_sweep cluster queue (read-only; none if absent)."""
    if not Path(queue_db).exists():
        return set()
    conn = sqlite3.connect(f"file:{queue_db}?mode=ro", uri=True, timeout=30)
    try:
        rows = conn.execute(
            "SELECT DISTINCT model_name FROM units WHERE status IN ('pending','running')"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _load_history(path: Path) -> dict[str, list[int]]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def _save_history(path: Path, history: dict[str, list[int]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="slurm_job_manager.reopen_eval",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=os.environ.get("SJM_DB"))
    ap.add_argument("--models-root", type=Path,
                    default=Path(os.environ.get("MODELS_ROOT", REPO_ROOT / "results" / "models")))
    ap.add_argument("--model", action="append", default=[],
                    help="limit to this model (repeatable)")
    ap.add_argument("--all", action="store_true", help="with --apply: re-open every incomplete model")
    ap.add_argument("--apply", action="store_true", help="actually re-open (default: dry run)")
    ap.add_argument("--max-reopens", type=int, default=2,
                    help="with --all: skip a model already re-opened this many times (default 2)")
    ap.add_argument("--history", type=Path, default=None,
                    help="re-open history JSON (default: reopen_eval_history.json next to the DB)")
    ap.add_argument("--aa-queue-db", type=Path, default=None,
                    help="aa_sweep cluster queue DB (default: aa_sweep.config.CLUSTER_QUEUE_DB)")
    args = ap.parse_args(argv)
    if not args.db:
        ap.error("no DB path (set --db or $SJM_DB)")
    if args.apply and not (args.all or args.model):
        ap.error("--apply needs --all or at least one --model")

    from aa_sweep import config as aa_config
    from aa_sweep.plan import load_excludes

    norms, eps_inputs, kinds = configured_grid()
    excludes = load_excludes(aa_config.EXCLUDES_CSV)
    aa_busy = aa_queue_active_models(args.aa_queue_db or Path(aa_config.CLUSTER_QUEUE_DB))
    history_path = args.history or Path(args.db).with_name("reopen_eval_history.json")
    history = _load_history(history_path)
    db = JobDB(args.db)
    wanted = set(args.model)
    jobs = [j for j in db.all_jobs()
            if j.status == STATUS_FINISHED and j.current_epoch >= j.total_epochs
            and (not wanted or j.model_name in wanted)]

    print(f"grid: {','.join(norms)} x eps {','.join(f'{e:g}' for e in eps_inputs)} "
          f"on {','.join(kinds)}  ({len(jobs)} finished rows checked)")
    incomplete, capped, busy = [], [], []
    for job in sorted(jobs, key=lambda j: j.model_name):
        if job.model_name in excludes:
            print(f"  {job.model_name}: excluded ({excludes[job.model_name]}), skipped", file=sys.stderr)
            continue
        if job.model_name in aa_busy:
            busy.append(job.model_name)
            print(f"  {job.model_name}: aa_sweep queue still has units for it, skipped", file=sys.stderr)
            continue
        model_dir = args.models_root / job.model_name
        if not model_dir.is_dir():
            print(f"  {job.model_name}: model dir missing ({model_dir}), skipped", file=sys.stderr)
            continue
        missing = missing_cells(model_dir, norms, eps_inputs, kinds)
        if not missing:
            history.pop(job.model_name, None)   # complete now: forget past re-opens
            continue
        detail = ", ".join(f"{k}={n}" for k, n in missing.items())
        n_prev = len(history.get(job.model_name, []))
        if not wanted and n_prev >= args.max_reopens:
            capped.append(job.model_name)
            print(f"  {job.model_name:55s} missing cells: {detail}  CAPPED "
                  f"(re-opened {n_prev}x already; investigate its final_eval log)")
            continue
        incomplete.append(job.model_name)
        print(f"  {job.model_name:55s} missing cells: {detail}"
              + (f"  (re-opened {n_prev}x before)" if n_prev else ""))

    for name in sorted(wanted - {j.model_name for j in jobs}):
        print(f"  {name}: not a finished fully-trained row, skipped", file=sys.stderr)

    print(f"{len(incomplete)} model(s) with an incomplete AA grid")
    reopened = 0
    if not args.apply:
        if incomplete:
            print("dry run -- re-run with --apply --all (or --model NAME) to re-open")
    else:
        now = int(time.time())
        for name in incomplete:
            if db.reopen_for_eval(name):
                reopened += 1
                history.setdefault(name, []).append(now)
        print(f"re-opened {reopened} row(s) -> pending (eval-only rerun on next claim)")
        _save_history(history_path, history)
    print(f"[reopen_eval] {'' if args.apply else 'DRY-RUN '}summary incomplete={len(incomplete)} "
          f"reopened={reopened} capped={len(capped)} aa_queue_busy={len(busy)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
