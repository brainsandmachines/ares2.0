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


def test_release_is_a_no_op_for_a_non_owner(db):
    # The collision bug: a task whose claim was taken over must not free the row
    # that the new owner is training.
    _seed(db, "a")
    db.claim_next(1001, {})
    assert db.release("a", owner=1002) == 0
    j = db.get("a")
    assert j.status == "running" and j.slurm_job_id == 1001 and j.requeued == 0
    # ...while the real owner still can.
    assert db.release("a", owner=1001) == 1
    assert db.get("a").status == "pending"


def test_requeue_dead_skips_freshly_claimed_rows(db):
    # A job id seconds old is not in squeue yet; releasing it hands the model to
    # a second trainer.
    _seed(db, "a")
    db.claim_next(111, {})
    assert db.requeue_dead(lambda jid: False, min_age_s=300) == 0
    assert db.get("a").status == "running"
    assert db.requeue_dead(lambda jid: False, min_age_s=0) == 1
    assert db.get("a").status == "pending"


def test_requeue_dead_only_frees_dead_owners(db):
    _seed(db, "alive", priority=1)
    _seed(db, "dead", priority=2)
    db.claim_next(111, {})   # alive
    db.claim_next(222, {})   # dead
    freed = db.requeue_dead(lambda jid: jid == 111, min_age_s=0)  # only 111 alive
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


# ---- eval lane ---------------------------------------------------------------
def _trained(db, name, epochs=150, owner=1):
    _seed(db, name, epochs=epochs)
    assert db.claim_next(owner, {}).model_name == name
    db.update_epoch(name, epochs)


def test_mark_trained_hands_row_to_eval_lane(db):
    _trained(db, "a")
    assert db.mark_trained("a") == 1
    j = db.get("a")
    assert j.status == "eval" and j.slurm_job_id is None and j.claimed_ts is None
    assert db.claim_next(2, {}) is None                     # train lane: nothing
    e = db.claim_next(3, {}, lane="eval")
    assert e.model_name == "a" and e.status == "running" and e.slurm_job_id == 3
    assert db.claim_next(4, {}, lane="eval") is None        # owned now


def test_eval_lane_ignores_pending_and_dependency_gate(db):
    _trained(db, "child")
    db.mark_trained("child")
    _seed(db, "fresh")
    # The dependency is not finished, but an eval row has already trained from it.
    assert db.claim_next(5, {"child": "parent_not_finished"}, lane="eval").model_name == "child"
    assert db.claim_next(6, {}, lane="eval") is None        # 'fresh' is the train lane's


def test_train_lane_skips_fully_trained_pending_rows(db):
    _trained(db, "done")
    conn = sqlite3.connect(db.db_path)
    conn.execute("UPDATE jobs SET status='pending', slurm_job_id=NULL WHERE model_name='done'")
    conn.commit()
    conn.close()
    _seed(db, "todo", priority=500)
    assert db.claim_next(2, {}).model_name == "todo"
    assert db.promote_trained(dry_run=True) == ["done"]
    assert db.get("done").status == "pending"
    assert db.promote_trained() == ["done"]
    assert db.get("done").status == "eval"


def test_release_sends_fully_trained_rows_to_eval(db):
    _seed(db, "mid")
    db.claim_next(1, {})
    db.update_epoch("mid", 70)
    _trained(db, "end", owner=2)
    assert db.release("mid", owner=1) == 1 and db.get("mid").status == "pending"
    assert db.release("end", owner=2) == 1 and db.get("end").status == "eval"
    # An eval-lane owner released at its time limit goes back to eval too.
    db.claim_next(3, {}, lane="eval")
    assert db.release("end", owner=3) == 1
    assert db.get("end").status == "eval" and db.get("end").requeued == 2


def test_reconcile_sends_unclaimed_eval_row_back_to_training(db):
    _trained(db, "a", epochs=100)
    db.mark_trained("a")
    assert db.reconcile("a", 100) == 0 and db.get("a").status == "eval"
    assert db.reconcile("a", 200) == 1 and db.get("a").status == "pending"


def test_unowned_eval_lists_only_claimable_eval_rows(db):
    _trained(db, "a", owner=1)
    db.mark_trained("a")
    _trained(db, "b", owner=2)
    db.mark_trained("b")
    db.claim_next(9, {}, lane="eval")
    assert len(db.unowned_eval()) == 1
