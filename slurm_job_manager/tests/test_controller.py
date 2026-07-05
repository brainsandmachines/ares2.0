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


def test_success_marks_finished(db, monkeypatch):
    db.upsert_pending("a", 150, 10)
    _run_with(monkeypatch, 0, "")
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)
    rc = controller.run_once(db, _rows("a"), {}, Path("/models"), 555)
    assert rc == 0
    assert db.get("a").status == "finished"


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
    controller.run_once(db, _rows("a"), {}, Path("/models"), 222)
    assert db.get("a").status == "finished"


def test_no_work_exits_zero(db, monkeypatch):
    _run_with(monkeypatch, 0, "")
    monkeypatch.setattr(controller, "_default_slurm_active", lambda jid: True)
    assert controller.run_once(db, {}, {}, Path("/models"), 1) == 0
