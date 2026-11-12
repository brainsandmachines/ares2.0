"""Offline tests for the controller's claim -> run -> outcome cycle.

``lifecycle.run`` and the Slurm-liveness probe are stubbed so no GPU/Slurm is
needed; we assert the DB transition the controller makes for each run outcome.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slurm_job_manager import controller
from slurm_job_manager.db import JobDB


@pytest.fixture
def db(tmp_path):
    return JobDB(str(tmp_path / "jobs.sqlite"))


def _rows(name):
    return {name: {"model_name": name, "init_mode": "scratch"}}


def _run_with(monkeypatch, rc, tail):
    monkeypatch.setattr(controller.lifecycle, "run", lambda *a, **k: (rc, tail))


@pytest.fixture(autouse=True)
def top_ups(monkeypatch):
    """Stub the eval-lane sbatch top-up (no Slurm here); record each call."""
    calls = []
    monkeypatch.setattr(controller.eval_lane, "top_up",
                        lambda db, **k: calls.append(db.db_path) or {"submitted": ["1"]})
    return calls


def _train(db, name, epochs=150):
    """Seed + let the stubbed run 'train' it to the last epoch (as the hook would)."""
    db.upsert_pending(name, epochs, 10)

    def fake_run(row, models_root, db_, **k):
        assert k["extra_overrides"] == ["final_eval=False"]
        db_.update_epoch(row["model_name"], epochs)
        return 0, ""
    return fake_run


def test_train_lane_success_goes_to_eval_and_queues_an_eval_job(db, monkeypatch, top_ups):
    monkeypatch.setattr(controller.lifecycle, "run", _train(db, "a"))
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)
    rc = controller.run_once(db, _rows("a"), {}, Path("/models"), 555)
    assert rc == 0
    j = db.get("a")
    assert j.status == "eval" and j.slurm_job_id is None
    assert len(top_ups) == 1


def test_train_lane_submit_failure_still_leaves_eval(db, monkeypatch):
    monkeypatch.setattr(controller.lifecycle, "run", _train(db, "a"))
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)

    def boom(db, **k):
        raise RuntimeError("sbatch: command not found")
    monkeypatch.setattr(controller.eval_lane, "top_up", boom)
    assert controller.run_once(db, _rows("a"), {}, Path("/models"), 555) == 0
    assert db.get("a").status == "eval"


def test_eval_lane_claims_eval_row_and_marks_finished(db, monkeypatch, top_ups):
    monkeypatch.setattr(controller.lifecycle, "run", _train(db, "a"))
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)
    controller.run_once(db, _rows("a"), {}, Path("/models"), 555)
    seen = {}

    def fake_eval(row, models_root, db_, **k):
        seen["overrides"] = k["extra_overrides"]
        seen["owner"] = db_.get("a").slurm_job_id
        return 0, ""
    monkeypatch.setattr(controller.lifecycle, "run", fake_eval)
    monkeypatch.setenv("SJM_EXTRA_OVERRIDES", "final_eval_aa_batch_size=32 final_eval_aa_max_batches=32")
    # The train lane finds nothing left: the trained row belongs to the eval lane.
    assert controller.run_once(db, _rows("a"), {}, Path("/models"), 600) == 0
    assert db.get("a").status == "eval"
    assert controller.run_once(db, _rows("a"), {}, Path("/models"), 777, lane="eval") == 0
    assert seen == {"overrides": ["final_eval_aa_batch_size=32", "final_eval_aa_max_batches=32"],
                    "owner": 777}
    assert db.get("a").status == "finished"
    assert len(top_ups) == 1                       # the eval lane queues nothing


def test_eval_lane_time_limit_release_returns_row_to_eval(db, monkeypatch):
    monkeypatch.setattr(controller.lifecycle, "run", _train(db, "a"))
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)
    controller.run_once(db, _rows("a"), {}, Path("/models"), 555)

    def killed(row, models_root, db_, **k):
        db_.release(row["model_name"], owner=777)    # the SIGTERM trap fired
        return 143, "CANCELLED DUE TO TIME LIMIT"
    monkeypatch.setattr(controller.lifecycle, "run", killed)
    controller.run_once(db, _rows("a"), {}, Path("/models"), 777, lane="eval")
    j = db.get("a")
    assert j.status == "eval" and j.slurm_job_id is None and j.requeued == 1


def test_transient_error_requeues(db, monkeypatch):
    db.upsert_pending("a", 150, 10)
    _run_with(monkeypatch, 1, "CANCELLED DUE TO TIME LIMIT")
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)
    controller.run_once(db, _rows("a"), {}, Path("/models"), 555)
    j = db.get("a")
    assert j.status == "pending" and j.slurm_job_id is None and j.requeued == 1


def test_deterministic_error_marks_failed_with_signature(db, monkeypatch):
    db.upsert_pending("a", 150, 10)
    _run_with(monkeypatch, 1, "Traceback (most recent call last):\nValueError: nope")
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)
    rc = controller.run_once(db, _rows("a"), {}, Path("/models"), 555)
    assert rc == 1
    j = db.get("a")
    assert j.status == "failed" and j.last_error_hash and "ValueError" in (j.last_error or "")


def test_requeue_dead_runs_before_claim(db, monkeypatch):
    # A dead-owner running row is freed, then claimed fresh by this task.
    db.upsert_pending("a", 150, 10)
    db.claim_next(111, {})            # owned by (soon-dead) job 111
    _run_with(monkeypatch, 0, "")
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: jid != 111)
    monkeypatch.setattr(controller, "REQUEUE_MIN_AGE_S", 0)
    controller.run_once(db, _rows("a"), {}, Path("/models"), 222)
    assert db.get("a").status == "eval"


def test_no_work_exits_zero(db, monkeypatch):
    _run_with(monkeypatch, 0, "")
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)
    assert controller.run_once(db, {}, {}, Path("/models"), 1) == 0


def test_liveness_fails_open_when_both_probes_error(monkeypatch):
    # A squeue/sacct timeout must never read as "the owner is dead".
    monkeypatch.setattr(controller, "_in_squeue", lambda jid: None)
    monkeypatch.setattr(controller, "_sacct_active", lambda jid: None)
    assert controller._default_slurm_active(1) is True
    # An answered "not active" still releases.
    monkeypatch.setattr(controller, "_in_squeue", lambda jid: False)
    monkeypatch.setattr(controller, "_sacct_active", lambda jid: False)
    assert controller._default_slurm_active(1) is False
    # squeue unreachable but sacct says active -> alive.
    monkeypatch.setattr(controller, "_in_squeue", lambda jid: None)
    monkeypatch.setattr(controller, "_sacct_active", lambda jid: True)
    assert controller._default_slurm_active(1) is True
