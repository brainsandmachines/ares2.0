"""Offline tests for JobDB.cron_release and the cron_release Slurm probe."""

from __future__ import annotations

import sqlite3
import subprocess

import pytest

from slurm_job_manager import cron_release as cr
from slurm_job_manager.db import JobDB

NOW = 1_800_000_000
HOUR = 3600
STARTED = NOW - 11 * HOUR  # owner start, before the default claim at NOW - 10h


@pytest.fixture
def db(tmp_path):
    return JobDB(str(tmp_path / "jobs.sqlite"))


def _running(db, name, job_id, claimed_ago=10 * HOUR, hb_ago=60, epoch=5, epochs=150):
    db.upsert_pending(name, epochs, 100)
    assert db.claim_next(job_id, {}).model_name == name
    conn = sqlite3.connect(db.db_path)
    conn.execute(
        "UPDATE jobs SET claimed_ts=?, heartbeat_ts=?, current_epoch=? WHERE model_name=?",
        (NOW - claimed_ago, NOW - hb_ago, epoch, name),
    )
    conn.commit()
    conn.close()


def _sweep(db, states, cancelled=None, **kw):
    cancelled = [] if cancelled is None else cancelled

    def cancel(job_id):
        cancelled.append(job_id)
        return True

    kw.setdefault("stall_s", 6 * HOUR)
    kw.setdefault("max_stalled", 2)
    return db.cron_release(states.get, cancel, now=NOW, **kw)


def test_live_owner_is_left_alone(db):
    _running(db, "a", 1)
    cancelled = []
    report = _sweep(db, {1: ("RUNNING", STARTED)}, cancelled)
    assert report.released == [] and cancelled == []
    assert db.get("a").status == "running" and db.get("a").slurm_job_id == 1


@pytest.mark.parametrize("state", ["", "CANCELLED", "COMPLETING", "COMPLETED", "NODE_FAIL"])
def test_owner_not_running_is_released(db, state):
    _running(db, "a", 1)
    report = _sweep(db, {1: (state, STARTED)})
    assert report.released == [("a", 1, state or "GONE")]
    row = db.get("a")
    assert row.status == "pending" and row.slurm_job_id is None and row.requeued == 1


def test_node_fail_requeue_pending_under_same_id_is_released(db):
    _running(db, "a", 1)
    report = _sweep(db, {1: ("PENDING", None)})
    assert report.released == [("a", 1, "PENDING")]
    assert db.get("a").status == "pending"


def test_restarted_incarnation_releases_only_its_old_row(db):
    _running(db, "old", 1, claimed_ago=10 * HOUR)
    _running(db, "new", 1, claimed_ago=HOUR - 60)  # same id, claimed after the restart
    report = _sweep(db, {1: ("RUNNING", NOW - HOUR)})
    assert report.released == [("old", 1, "RESTARTED")]
    assert db.get("old").status == "pending"
    assert db.get("new").status == "running" and db.get("new").slurm_job_id == 1


def test_probe_failure_fails_open(db):
    _running(db, "a", 1, hb_ago=20 * HOUR)
    cancelled = []
    report = _sweep(db, {}, cancelled)
    assert report.unknown == [("a", 1)]
    assert report.released == [] and cancelled == []
    assert db.get("a").status == "running"


def test_fresh_claim_is_skipped(db):
    _running(db, "a", 1, claimed_ago=60, hb_ago=60)
    report = _sweep(db, {1: ("", None)})
    assert report.released == []
    assert db.get("a").status == "running"


def test_stalled_owner_is_cancelled_not_released(db):
    _running(db, "a", 1, hb_ago=7 * HOUR)
    cancelled = []
    report = _sweep(db, {1: ("RUNNING", STARTED)}, cancelled)
    assert cancelled == [1]
    assert report.cancelled == [("a", 1, 7 * HOUR)]
    assert db.get("a").status == "running" and db.get("a").slurm_job_id == 1


def test_heartbeat_below_threshold_is_not_cancelled(db):
    _running(db, "a", 1, hb_ago=5 * HOUR)
    cancelled = []
    report = _sweep(db, {1: ("RUNNING", STARTED)}, cancelled)
    assert report.stalled == [] and cancelled == []


def test_final_eval_is_never_cancelled(db):
    _running(db, "a", 1, hb_ago=20 * HOUR, epoch=150, epochs=150)
    cancelled = []
    report = _sweep(db, {1: ("RUNNING", STARTED)}, cancelled)
    assert report.stalled == [] and cancelled == []


def test_breaker_blocks_cancels_when_many_stall_at_once(db):
    for i in range(4):
        _running(db, f"m{i}", 100 + i, hb_ago=7 * HOUR)
    cancelled = []
    report = _sweep(db, {100 + i: ("RUNNING", STARTED) for i in range(4)}, cancelled,
                    max_stalled=3)
    assert report.breaker_tripped
    assert len(report.stalled) == 4 and report.cancelled == [] and cancelled == []


def test_breaker_does_not_block_dead_owner_releases(db):
    for i in range(4):
        _running(db, f"m{i}", 100 + i, hb_ago=7 * HOUR)
    _running(db, "dead", 200)
    states = {100 + i: ("RUNNING", STARTED) for i in range(4)}
    states[200] = ("CANCELLED", STARTED)
    report = _sweep(db, states, max_stalled=3)
    assert report.breaker_tripped
    assert report.released == [("dead", 200, "CANCELLED")]


def test_dry_run_writes_nothing(db):
    _running(db, "dead", 1)
    _running(db, "hung", 2, hb_ago=7 * HOUR)
    cancelled = []
    report = _sweep(db, {1: ("", None), 2: ("RUNNING", STARTED)}, cancelled, dry_run=True)
    assert report.released == [("dead", 1, "GONE")]
    assert report.cancelled == [("hung", 2, 7 * HOUR)]
    assert cancelled == []
    assert db.get("dead").status == "running" and db.get("hung").status == "running"


def test_release_claimed_ts_guard(db):
    _running(db, "a", 1)
    claimed = db.get("a").claimed_ts
    assert db.release("a", owner=1, claimed_ts=claimed + 1) == 0
    assert db.release("a", owner=1, claimed_ts=claimed) == 1


# ---- probe ----------------------------------------------------------------------
def _fake_run(squeue, sacct):
    def run(argv):
        return {"squeue": squeue, "sacct": sacct}[argv[0]]
    return run


def _cp(rc, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


def test_probe_uses_squeue_when_it_lists_the_job(monkeypatch):
    monkeypatch.setattr(cr, "_run", _fake_run(_cp(0, "RUNNING|1788955891\n"), None))
    assert cr.owner_state(1) == ("RUNNING", 1788955891)


def test_probe_pending_has_no_start(monkeypatch):
    monkeypatch.setattr(cr, "_run", _fake_run(_cp(0, "PENDING|N/A\n"), None))
    assert cr.owner_state(1) == ("PENDING", None)


def test_probe_falls_back_to_sacct_for_a_job_gone_from_squeue(monkeypatch):
    sq = _cp(1, "", "slurm_load_jobs error: Invalid job id specified")
    monkeypatch.setattr(cr, "_run", _fake_run(sq, _cp(0, "CANCELLED by 123|1789030264\n")))
    assert cr.owner_state(1) == ("CANCELLED", 1789030264)


def test_probe_reports_gone_when_neither_side_knows_the_id(monkeypatch):
    sq = _cp(1, "", "slurm_load_jobs error: Invalid job id specified")
    monkeypatch.setattr(cr, "_run", _fake_run(sq, _cp(0, "")))
    assert cr.owner_state(1) == ("", None)


def test_probe_fails_open_when_slurm_is_unreachable(monkeypatch):
    sq = _cp(1, "", "slurm_load_jobs error: Unable to contact slurm controller")
    monkeypatch.setattr(cr, "_run", _fake_run(sq, _cp(0, "")))
    assert cr.owner_state(1) is None
    monkeypatch.setattr(cr, "_run", _fake_run(None, None))
    assert cr.owner_state(1) is None
