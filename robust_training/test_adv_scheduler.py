import sqlite3
import time
import tempfile
import os
import pytest
import multiprocessing
import threading

from adv_scheduler import TaskScheduler  


@pytest.fixture
def db_scheduler():
    """Create a temporary SQLite DB with the same schema for testing."""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    path = tmp.name
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE models (
            model_id INTEGER PRIMARY KEY,
            norm TEXT,
            constraint_val REAL,
            adv_train INTEGER,
            init_id INTEGER,
            current_epoch INTEGER DEFAULT 0,
            last_progress_ts INTEGER,
            last_time_selected INTEGER,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Initialize your scheduler instance
    sch = TaskScheduler(db_path=path)
    yield sch

    os.remove(path)


def insert_model(sch, **fields):
    """Helper to insert a model record."""
    conn = sqlite3.connect(sch.db_path)
    c = conn.cursor()
    defaults = dict(
        norm="Linf",
        constraint_val=8.0,
        adv_train=1,
        init_id=0,
        current_epoch=0,
        last_progress_ts=None,
        last_time_selected=None,
        status="waiting",
    )
    defaults.update(fields)
    c.execute("""
        INSERT INTO models (norm, constraint_val, adv_train, init_id,
                            current_epoch, last_progress_ts, last_time_selected, status)
        VALUES (:norm, :constraint_val, :adv_train, :init_id,
                :current_epoch, :last_progress_ts, :last_time_selected, :status)
    """, defaults)
    model_id = c.lastrowid
    conn.commit()
    conn.close()
    return model_id


# ---- Tests ----

def test_update_progress_epoch_end_increments_epoch_and_updates_timestamp(db_scheduler):
    model_id = insert_model(db_scheduler, status="training", current_epoch=5)
    db_scheduler.update_progress_epoch_end(model_id, max_epoch=10)

    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT current_epoch, last_progress_ts, status FROM models WHERE model_id=?", (model_id,))
    epoch, ts, status = c.fetchone()
    conn.close()

    assert epoch == 6
    assert isinstance(ts, int)
    assert status == "training"


def test_update_progress_epoch_end_sets_finished_when_max_epoch_reached(db_scheduler):
    model_id = insert_model(db_scheduler, status="training", current_epoch=9)
    db_scheduler.update_progress_epoch_end(model_id, max_epoch=10)

    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT status FROM models WHERE model_id=?", (model_id,))
    (status,) = c.fetchone()
    conn.close()

    assert status == "finished"


def test_requeue_stale_trainings_sets_stale_training_to_waiting(db_scheduler):
    old_ts = int(time.time()) - 11 * 3600  # 11 hours ago
    model_id = insert_model(db_scheduler, status="training", last_progress_ts=old_ts, current_epoch=50)
    affected = db_scheduler.requeue_stale_trainings(threshold_hours=10, max_epoch=200)

    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT status FROM models WHERE model_id=?", (model_id,))
    (status,) = c.fetchone()
    conn.close()

    assert affected == 1
    assert status == "waiting"


def test_claim_next_waiting_model_sets_training_and_updates_last_time_selected(db_scheduler):
    model_id = insert_model(db_scheduler, status="waiting")
    claimed = db_scheduler.claim_next_waiting_model()

    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT status, last_time_selected FROM models WHERE model_id=?", (model_id,))
    status, last_time_selected = c.fetchone()
    conn.close()

    assert claimed is not None
    assert status == "training"
    assert isinstance(last_time_selected, int)
    
# ---- Concurrency / locking tests ----

def _claim_in_subprocess(db_path, return_dict, key):
    """Helper: run claim_next_waiting_model() in a subprocess."""
    from adv_scheduler import TaskScheduler
    sch = TaskScheduler(db_path)
    start = time.time()
    try:
        result = sch.claim_next_waiting_model(cooldown_minutes=0)
        return_dict[key] = {
            "result": result,
            "duration": time.time() - start,
        }
    except Exception as e:
        return_dict[key] = {
            "result": None,
            "duration": time.time() - start,
            "error": str(e),
        }


def test_concurrent_multi_claims_no_lock_or_duplicate(db_scheduler):
    """
    Ensure multiple processes can concurrently claim models without lockups
    and each gets a unique model.
    """
    # Insert 5 waiting models
    for i in range(5):
        insert_model(db_scheduler, status="waiting", norm="l2", constraint_val=i + 1)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []

    # Launch 5 parallel claimers
    for i in range(5):
        p = multiprocessing.Process(
            target=_claim_in_subprocess,
            args=(db_scheduler.db_path, return_dict, f"p{i+1}")
        )
        procs.append(p)
        p.start()

    # Join with timeout — fail if any process hangs
    for p in procs:
        p.join(timeout=15)
        assert not p.is_alive(), "One of the processes hung (possible DB lock)."

    results = list(return_dict.values())

    # Ensure all succeeded
    successes = [r["result"] for r in results if r and r["result"]]
    assert len(successes) == 5, f"Expected 5 models claimed, got {len(successes)}"

    # All model_ids must be unique
    model_ids = [r["model_id"] for r in successes]
    assert len(set(model_ids)) == 5, f"Duplicate model IDs claimed: {model_ids}"

    # Each claim should be fast (<10s)
    for r in results:
        assert r["duration"] < 10, f"Claim took too long: {r['duration']:.2f}s"

    # Verify DB consistency
    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM models WHERE status='training'")
    (training_count,) = c.fetchone()
    conn.close()

    assert training_count == 5
    print(f"\n[PASS] {training_count} models claimed concurrently with no DB locks.")

def test_requeue_and_claim_do_not_deadlock(db_scheduler):
    """Ensure requeue_stale_trainings and claim_next_waiting_model can run concurrently without freezing."""

    # Insert one training and one waiting model
    m1 = insert_model(db_scheduler, status="training", current_epoch=5,
                      last_progress_ts=int(time.time()) - 11 * 3600)
    m2 = insert_model(db_scheduler, status="waiting", current_epoch=0)

    def run_requeue():
        db_scheduler.requeue_stale_trainings(threshold_hours=10, max_epoch=200)

    def run_claim(result_holder):
        model = db_scheduler.claim_next_waiting_model()
        result_holder["claimed"] = model

    result = {}

    t1 = threading.Thread(target=run_requeue)
    t2 = threading.Thread(target=run_claim, args=(result,))

    t1.start()
    time.sleep(0.1)  # slight overlap
    t2.start()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive(), "requeue_stale_trainings got stuck!"
    assert not t2.is_alive(), "claim_next_waiting_model got stuck!"
    assert result["claimed"] is not None, "claim_next_waiting_model failed to get a model"