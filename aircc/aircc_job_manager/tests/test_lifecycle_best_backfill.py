"""Unit tests for lifecycle.ensure_best_checkpoint (parent-side backfill).

Guards the failure seen on 2026-08-02/03: the in-training hook's single
best-checkpoint write hit a transient sqlite CANTOPEN, swallowed it, and left
four finished rows with a NULL best_checkpoint -- which silently gated the
continuation rows depending on them.
"""

from __future__ import annotations

import json

from aircc.aircc_job_manager.db import AirccDB
from aircc.aircc_job_manager.lifecycle import ensure_best_checkpoint


def _model_dir(tmp_path, name, norm="l1", eps=1.0):
    models_root = tmp_path / "models"
    d = models_root / name
    d.mkdir(parents=True)
    (d / "model_best.pth.tar").write_bytes(b"best")
    (d / "last.pth.tar").write_bytes(b"last")
    (d / "autoattack_eps_norm_scores.json").write_text(
        json.dumps({
            "attack_norm": norm,
            "epsilon_input": eps,
            "scores": {"model_best.pth.tar": 40.0, "last.pth.tar": 55.5},
        })
    )
    return models_root


def test_backfills_when_in_training_hook_left_it_null(tmp_path):
    models_root = _model_dir(tmp_path, "A")
    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    db.upsert_pending("A", 200, 0)
    row = {"model_name": "A", "threat_norm": "l1", "threat_eps": "1"}

    ensure_best_checkpoint(row, models_root, db, log=lambda *_: None)

    job = db.get("A")
    assert job.best_checkpoint == str((models_root / "A" / "last.pth.tar").resolve())
    assert job.best_score == 55.5


def test_does_not_overwrite_what_the_hook_already_wrote(tmp_path):
    models_root = _model_dir(tmp_path, "A")
    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    db.upsert_pending("A", 200, 0)
    db.set_best_checkpoint("A", "/from/hook.pth.tar", 12.5)
    row = {"model_name": "A", "threat_norm": "l1", "threat_eps": "1"}

    ensure_best_checkpoint(row, models_root, db, log=lambda *_: None)

    job = db.get("A")
    assert job.best_checkpoint == "/from/hook.pth.tar"
    assert job.best_score == 12.5


def test_never_raises_when_nothing_is_scorable(tmp_path):
    models_root = tmp_path / "models"
    (models_root / "A").mkdir(parents=True)  # no AA output at all
    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    db.upsert_pending("A", 200, 0)
    row = {"model_name": "A", "threat_norm": "l1", "threat_eps": "1"}

    ensure_best_checkpoint(row, models_root, db, log=lambda *_: None)  # must not raise

    assert db.get("A").best_checkpoint is None


def test_never_raises_when_the_db_write_fails(tmp_path):
    """The whole point is resilience: a second failure must still let the row finish."""
    models_root = _model_dir(tmp_path, "A")
    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    db.upsert_pending("A", 200, 0)

    def boom(*_a, **_k):
        raise RuntimeError("unable to open database file")

    db.set_best_checkpoint = boom
    row = {"model_name": "A", "threat_norm": "l1", "threat_eps": "1"}

    ensure_best_checkpoint(row, models_root, db, log=lambda *_: None)  # must not raise


def test_unblocks_a_dependent_continuation_row(tmp_path):
    """End-to-end of the actual outage: dep finished with NULL best -> child gated."""
    models_root = _model_dir(tmp_path, "dep", norm="l2", eps=4.0)
    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    db.upsert_pending("dep", 200, 0)
    db.upsert_pending("child", 40, 1)
    deps = {"dep": "", "child": "dep"}

    db.claim_next(1, deps)
    db.mark_finished("dep")  # hook's write was swallowed -> best_checkpoint still NULL
    assert db.claim_next(2, deps) is None

    ensure_best_checkpoint(
        {"model_name": "dep", "threat_norm": "l2", "threat_eps": "4"},
        models_root, db, log=lambda *_: None,
    )

    assert db.claim_next(2, deps).model_name == "child"
