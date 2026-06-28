"""Unit tests for AirccDB: atomic claiming, dependency gating, stale requeue."""

from __future__ import annotations

import threading
import time

import pytest

from aircc.aircc_job_manager.db import AirccDB


@pytest.fixture()
def db(tmp_path):
    return AirccDB(str(tmp_path / "jobs.sqlite"))


def test_dependency_gate(db):
    db.upsert_pending("A", 150, 0)
    db.upsert_pending("B", 40, 1)
    deps = {"A": "", "B": "A"}

    assert db.claim_next(1, deps).model_name == "A"      # no dep -> claimable
    assert db.claim_next(2, deps) is None                # B blocked (A not finished)

    db.mark_finished("A")
    assert db.claim_next(2, deps) is None                # A finished but no best ckpt

    db.set_best_checkpoint("A", "/m/A/model_best.pth.tar", 55.0)
    assert db.claim_next(2, deps).model_name == "B"      # now satisfied


def test_no_double_claim_concurrent(db):
    # Seed many independent models; hammer claim from many threads; assert each
    # model is handed out at most once.
    n = 50
    for i in range(n):
        db.upsert_pending(f"m{i:03d}", 10, i)
    deps = {f"m{i:03d}": "" for i in range(n)}

    claimed: list[str] = []
    lock = threading.Lock()

    def worker(tid):
        while True:
            j = db.claim_next(tid, deps)
            if j is None:
                return
            with lock:
                claimed.append(j.model_name)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == n
    assert len(set(claimed)) == n        # no model claimed twice


def test_priority_order(db):
    db.upsert_pending("low", 10, 5)
    db.upsert_pending("high", 10, 1)
    db.upsert_pending("mid", 10, 3)
    order = [db.claim_next(1, {}).model_name for _ in range(3)]
    assert order == ["high", "mid", "low"]


def test_requeue_stale(db):
    db.upsert_pending("X", 10, 0)
    job = db.claim_next(1, {})
    assert job.model_name == "X" and db.get("X").status == "running"

    # Fresh heartbeat -> not stale.
    assert db.requeue_stale(stale_seconds=10_000) == 0
    assert db.get("X").status == "running"

    # Force staleness: backdate heartbeat by claiming threshold 0 after a tick.
    time.sleep(1)
    assert db.requeue_stale(stale_seconds=0) == 1
    assert db.get("X").status == "pending"
    assert db.get("X").owner_task is None
