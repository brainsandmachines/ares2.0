"""Best-checkpoint selection + DB link for the orchestrator plotting stage.

Skipped automatically if pandas/matplotlib aren't importable (e.g. a bare base
env); run under the `ares` conda env on botero where they are present.
"""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402
matplotlib.use("Agg")

# data_analysis is not a package; import the script module by path.
_DA = Path(__file__).resolve().parents[2] / "data_analysis"
sys.path.insert(0, str(_DA))
import plot_autoattack_comparation_orch as orchplot  # noqa: E402

from orchestrator.db import OrchestratorDB, STATUS_FINISHED  # noqa: E402

_CSV_BY_KIND = {
    "best": "autoattack_sweep_results.csv",
    "last": "autoattack_sweep_results_last.csv",
    "advbest": "autoattack_sweep_results_advbest.csv",
}


def _write_csv(path: Path, l2_eps2_robust: float, clean: float) -> None:
    rows = ["attack_norm,epsilon_input,clean_acc,robust_acc"]
    for norm in ("linf", "l2", "l1"):
        for eps in (1, 2, 4, 6, 8, 12, 16):
            robust = l2_eps2_robust if (norm == "l2" and eps == 2) else 10.0
            rows.append(f"{norm},{eps},{clean},{robust}")
    path.write_text("\n".join(rows) + "\n")


def test_best_checkpoint_by_robust_at_threat_model(tmp_path):
    name = "convnext_small_l2_2_init1"
    mdir = tmp_path / name
    mdir.mkdir()
    _write_csv(mdir / _CSV_BY_KIND["best"], l2_eps2_robust=40.0, clean=70.0)
    _write_csv(mdir / _CSV_BY_KIND["last"], l2_eps2_robust=55.0, clean=65.0)   # winner
    _write_csv(mdir / _CSV_BY_KIND["advbest"], l2_eps2_robust=50.0, clean=60.0)

    kind, score = orchplot.best_checkpoint_kind(tmp_path, name)
    assert kind == "last" and score == 55.0          # robust_acc @ l2 eps2


def test_best_checkpoint_clean_fallback_for_baseline(tmp_path):
    name = "convnext_small_baseline_init1"            # no norm/eps -> clean acc
    mdir = tmp_path / name
    mdir.mkdir()
    _write_csv(mdir / _CSV_BY_KIND["best"], l2_eps2_robust=10.0, clean=80.0)   # winner
    _write_csv(mdir / _CSV_BY_KIND["last"], l2_eps2_robust=10.0, clean=60.0)
    _write_csv(mdir / _CSV_BY_KIND["advbest"], l2_eps2_robust=10.0, clean=70.0)

    kind, score = orchplot.best_checkpoint_kind(tmp_path, name)
    assert kind == "best" and score == 80.0          # clean accuracy fallback


def test_no_norm_token_defaults_to_l2(tmp_path):
    # gradnorm_* names carry an eps but no norm token -> scored under l2.
    name = "convnext_small_gradnorm_2_init1"
    mdir = tmp_path / name
    mdir.mkdir()
    # l2 eps2 robust differs per kind; clean is high. If it scored clean it would
    # pick 'best' (clean 80); scoring l2@eps2 must pick 'last' (robust 55).
    _write_csv(mdir / _CSV_BY_KIND["best"], l2_eps2_robust=40.0, clean=80.0)
    _write_csv(mdir / _CSV_BY_KIND["last"], l2_eps2_robust=55.0, clean=60.0)   # winner under l2@2

    kind, score = orchplot.best_checkpoint_kind(tmp_path, name)
    assert kind == "last" and score == 55.0


def test_off_grid_eps_snaps_to_default(tmp_path):
    # eps 160 is not on the AA grid -> snaps to DEFAULT_EPS (4). The synthetic CSV
    # puts the distinguishing value at l2 eps2, so to prove the snap we make l2
    # eps4 the decider instead.
    name = "convnext_small_gradnorm_l2_160_init1"
    mdir = tmp_path / name
    mdir.mkdir()

    def _csv_eps4(path, l2_eps4, clean):
        rows = ["attack_norm,epsilon_input,clean_acc,robust_acc"]
        for norm in ("linf", "l2", "l1"):
            for eps in (1, 2, 4, 6, 8, 12, 16):
                robust = l2_eps4 if (norm == "l2" and eps == 4) else 99.0
                rows.append(f"{norm},{eps},{clean},{robust}")
        path.write_text("\n".join(rows) + "\n")

    _csv_eps4(mdir / _CSV_BY_KIND["best"], l2_eps4=10.0, clean=70.0)
    _csv_eps4(mdir / _CSV_BY_KIND["last"], l2_eps4=22.0, clean=70.0)            # winner at l2@4
    assert orchplot.DEFAULT_EPS == 4.0
    kind, score = orchplot.best_checkpoint_kind(tmp_path, name)
    assert kind == "last" and score == 22.0


def test_best_checkpoint_handles_missing_csvs(tmp_path):
    name = "convnext_small_l2_2_init1"
    mdir = tmp_path / name
    mdir.mkdir()
    _write_csv(mdir / _CSV_BY_KIND["last"], l2_eps2_robust=55.0, clean=65.0)   # only one
    kind, score = orchplot.best_checkpoint_kind(tmp_path, name)
    assert kind == "last" and score == 55.0


def test_finish_in_db_writes_status_and_best(tmp_path, monkeypatch):
    db_path = str(tmp_path / "o.db")
    db = OrchestratorDB(db_path)
    db.upsert_model("convnext_small_l2_2_init1", "golan-trainmodels",
                    "l2_2_init1", "/d/m", 0, 0, 250)
    db.claim_next("rtx6000", 1)
    db.force_status("convnext_small_l2_2_init1", "PLOTTING")

    monkeypatch.setenv("ORCH_MODEL_ID", "convnext_small_l2_2_init1")
    monkeypatch.setenv("ORCH_DB", db_path)
    orchplot._finish_in_db("last", 55.0)

    row = db.get_model_state("convnext_small_l2_2_init1")
    assert row.status == STATUS_FINISHED
    assert row.best_checkpoint == "last" and row.best_score == 55.0


def test_finish_in_db_noop_when_untracked(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_MODEL_ID", raising=False)
    monkeypatch.delenv("ORCH_DB", raising=False)
    orchplot._finish_in_db("best", 12.3)  # must not raise
