"""Offline unit tests for JobDB: claim, test lane, dep gate, release, requeue."""

from __future__ import annotations

import sqlite3

import pytest

from slurm_job_manager.db import JobDB


@pytest.fixture
def db(tmp_path):
    return JobDB(str(tmp_path / "jobs.sqlite"))


def _seed(db, name, epochs=150, priority=100, is_test=False):
    db.upsert_pending(name, epochs, priority, is_test=is_test)


def test_claim_is_atomic_no_double_claim(db):
    _seed(db, "a", priority=10)
    j1 = db.claim_next(1001, {})
    assert j1 is not None and j1.model_name == "a" and j1.status == "running"
    assert j1.slurm_job_id == 1001
    # No other pending row -> a second task claims nothing (the row is owned).
    assert db.claim_next(1002, {}) is None


def test_priority_order_lower_first(db):
    _seed(db, "hi", priority=5)
    _seed(db, "lo", priority=50)
    assert db.claim_next(1, {}).model_name == "hi"
    assert db.claim_next(2, {}).model_name == "lo"


def test_test_lane_claimed_before_production(db):
    _seed(db, "prod", priority=0)          # best production priority
    _seed(db, "smoke", priority=100, is_test=True)  # worse priority, but test lane
    assert db.claim_next(1, {}).model_name == "smoke"
    assert db.claim_next(2, {}).model_name == "prod"


def test_dependency_gate_blocks_until_source_finished(db):
    _seed(db, "src", priority=1)
    _seed(db, "child", priority=2)
    deps = {"child": "src", "src": ""}
    # child is blocked -> only src is claimable.
    assert db.claim_next(1, deps).model_name == "src"
    assert db.claim_next(2, deps) is None          # child still blocked (src running)
    # src finishes but has no best_checkpoint yet -> still blocked.
    db.mark_finished("src")
    assert db.claim_next(3, deps) is None
    # once src has a best checkpoint, child unblocks.
    db.set_best_checkpoint("src", "/models/src/model_best.pth.tar", 42.0)
    assert db.claim_next(4, deps).model_name == "child"


def test_release_returns_to_pending_and_bumps_requeued(db):
    _seed(db, "a")
    db.claim_next(1001, {})
    assert db.release("a") == 1
    j = db.get("a")
    assert j.status == "pending" and j.slurm_job_id is None and j.requeued == 1
    # releasing a finished row is a no-op.
    db.mark_finished("a")
    assert db.release("a") == 0


def test_requeue_dead_only_frees_dead_owners(db):
    _seed(db, "alive", priority=1)
    _seed(db, "dead", priority=2)
    db.claim_next(111, {})   # alive
    db.claim_next(222, {})   # dead
    freed = db.requeue_dead(lambda jid: jid == 111)  # only 111 is alive
    assert freed == 1
    assert db.get("dead").status == "pending"
    assert db.get("alive").status == "running"


def test_reconcile_reopens_finished_when_epochs_raised(db):
    _seed(db, "a", epochs=100)
    db.claim_next(1, {})
    db.update_epoch("a", 100)
    db.mark_finished("a")
    # same target -> no reopen; higher target -> reopen.
    assert db.reconcile("a", 100) == 0
    assert db.reconcile("a", 200) == 1
    assert db.get("a").status == "pending" and db.get("a").total_epochs == 200


def test_get_by_slurm_job_id(db):
    _seed(db, "a")
    db.claim_next(9090, {})
    assert db.get_by_slurm_job_id(9090).model_name == "a"
    assert db.get_by_slurm_job_id(1) is None


def test_failure_hash_dedup(db):
    _seed(db, "a")
    assert not db.has_failure_hash("h1")
    db.record_failure_hash("h1", "a", "/tmp/h1.md")
    assert db.has_failure_hash("h1")
    # idempotent insert.
    db.record_failure_hash("h1", "a", "/tmp/h1.md")


def test_aircc_progress_hooks_write_to_superset_schema(tmp_path):
    """The already-live aircc hooks (AirccDB) must write into JobDB's table."""
    path = str(tmp_path / "jobs.sqlite")
    db = JobDB(path)
    db.upsert_pending("a", 150, 10)
    aircc_db = pytest.importorskip("aircc.aircc_job_manager.db")
    a = aircc_db.AirccDB(path)          # opens the SAME file; CREATE IF NOT EXISTS is a no-op
    a.update_epoch("a", 7)
    a.set_best_checkpoint("a", "/models/a/model_best.pth.tar", 55.5)
    j = db.get("a")
    assert j.current_epoch == 7
    assert j.best_checkpoint.endswith("model_best.pth.tar")
    assert j.best_score == 55.5
    # and JobDB's own columns still resolve after aircc touched the table.
    assert db.claim_next(1, {}).slurm_job_id == 1
