"""Offline tests for the eval-only re-open path (db.reopen_for_eval + reopen_eval CLI)."""

from __future__ import annotations

import csv

import pytest

from slurm_job_manager import reopen_eval
from slurm_job_manager.db import JobDB

FIELDS = ["model_name", "checkpoint_path", "attack_norm", "epsilon_input", "robust_acc"]


@pytest.fixture
def db(tmp_path):
    return JobDB(str(tmp_path / "jobs.sqlite"))


def _finish(db, name, epochs=10, reached=10):
    db.upsert_pending(name, epochs, 1)
    db.claim_next(1, {})
    db.update_epoch(name, reached)
    db.mark_finished(name)


def _write_model(models_root, name, cells):
    """A model dir with best+last checkpoints; ``cells`` = (norm, eps) rows in both CSVs."""
    d = models_root / name
    d.mkdir(parents=True)
    for ckpt, csv_name in (("model_best.pth.tar", "autoattack_sweep_results.csv"),
                           ("last.pth.tar", "autoattack_sweep_results_last.csv")):
        (d / ckpt).write_bytes(b"x")
        with (d / csv_name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for norm, eps in cells:
                w.writerow({"model_name": d.name, "checkpoint_path": str(d / ckpt),
                            "attack_norm": norm, "epsilon_input": eps, "robust_acc": 1.0})
    return d


FULL = [(n, e) for n in ("linf", "l2", "l1") for e in (1, 2, 4, 6, 8)]


def test_reopen_for_eval_only_touches_finished_fully_trained(db):
    _finish(db, "done")
    _finish(db, "short", reached=7)
    db.upsert_pending("pend", 10, 1)

    assert db.reopen_for_eval("done") == 1
    assert db.get("done").status == "pending"
    assert db.get("done").slurm_job_id is None
    assert db.reopen_for_eval("short") == 0      # reconcile's job, not ours
    assert db.get("short").status == "finished"
    assert db.reopen_for_eval("pend") == 0
    assert db.reopen_for_eval("missing") == 0


def test_configured_grid_matches_config_yaml():
    norms, eps, kinds = reopen_eval.configured_grid()
    assert norms == ["linf", "l2", "l1"]
    assert eps == [1.0, 2.0, 4.0, 6.0, 8.0]
    assert kinds == ["best", "last", "advbest"]


def test_missing_cells_counts_per_existing_kind(tmp_path):
    d = _write_model(tmp_path, "m", [("l2", 4)])
    norms, eps, kinds = reopen_eval.configured_grid()
    # advbest has no checkpoint -> not reported.
    assert reopen_eval.missing_cells(d, norms, eps, kinds) == {"best": 14, "last": 14}
    full = _write_model(tmp_path, "full", FULL)
    assert reopen_eval.missing_cells(full, norms, eps, kinds) == {}


def test_cli_dry_run_writes_nothing_and_apply_all_reopens_incomplete(db, tmp_path, capsys):
    models = tmp_path / "models"
    _finish(db, "arch/partial")
    _finish(db, "complete")
    _write_model(models, "arch/partial", [("l2", 4)])
    _write_model(models, "complete", FULL)

    base = ["--db", str(tmp_path / "jobs.sqlite"), "--models-root", str(models)]
    assert reopen_eval.main(base) == 0
    out = capsys.readouterr().out
    listed = [line.split()[0] for line in out.splitlines() if "missing cells" in line]
    assert listed == ["arch/partial"]
    assert db.get("arch/partial").status == "finished"   # dry run

    with pytest.raises(SystemExit):
        reopen_eval.main(base + ["--apply"])               # needs --all/--model

    assert reopen_eval.main(base + ["--apply", "--all"]) == 0
    assert db.get("arch/partial").status == "pending"
    assert db.get("complete").status == "finished"


def test_cli_skips_sweep_excludes(db, tmp_path, monkeypatch, capsys):
    from aa_sweep import config as aa_config

    excludes = tmp_path / "excludes.csv"
    excludes.write_text("model_name,reason\nbad,catastrophic overfitting\n")
    monkeypatch.setattr(aa_config, "EXCLUDES_CSV", excludes)
    models = tmp_path / "models"
    _finish(db, "bad")
    _write_model(models, "bad", [("l2", 4)])

    base = ["--db", str(tmp_path / "jobs.sqlite"), "--models-root", str(models)]
    assert reopen_eval.main(base + ["--apply", "--all"]) == 0
    assert "0 model(s)" in capsys.readouterr().out
    assert db.get("bad").status == "finished"
