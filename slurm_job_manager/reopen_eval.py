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

Caveat: while a re-opened row is pending/running, its not-yet-claimed dependents
are blocked again (the claim gate wants the dependency finished).
"""

from __future__ import annotations

import argparse
import os
import sys
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
    args = ap.parse_args(argv)
    if not args.db:
        ap.error("no DB path (set --db or $SJM_DB)")
    if args.apply and not (args.all or args.model):
        ap.error("--apply needs --all or at least one --model")

    norms, eps_inputs, kinds = configured_grid()
    db = JobDB(args.db)
    wanted = set(args.model)
    jobs = [j for j in db.all_jobs()
            if j.status == STATUS_FINISHED and j.current_epoch >= j.total_epochs
            and (not wanted or j.model_name in wanted)]

    print(f"grid: {','.join(norms)} x eps {','.join(f'{e:g}' for e in eps_inputs)} "
          f"on {','.join(kinds)}  ({len(jobs)} finished rows checked)")
    incomplete = []
    for job in sorted(jobs, key=lambda j: j.model_name):
        model_dir = args.models_root / job.model_name
        if not model_dir.is_dir():
            print(f"  {job.model_name}: model dir missing ({model_dir}), skipped", file=sys.stderr)
            continue
        missing = missing_cells(model_dir, norms, eps_inputs, kinds)
        if missing:
            incomplete.append(job.model_name)
            detail = ", ".join(f"{k}={n}" for k, n in missing.items())
            print(f"  {job.model_name:55s} missing cells: {detail}")

    for name in sorted(wanted - {j.model_name for j in jobs}):
        print(f"  {name}: not a finished fully-trained row, skipped", file=sys.stderr)

    print(f"{len(incomplete)} model(s) with an incomplete AA grid")
    if not args.apply:
        if incomplete:
            print("dry run -- re-run with --apply --all (or --model NAME) to re-open")
        return 0

    reopened = sum(db.reopen_for_eval(name) for name in incomplete)
    print(f"re-opened {reopened} row(s) -> pending (eval-only rerun on next claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
