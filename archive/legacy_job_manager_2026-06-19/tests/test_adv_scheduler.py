import sqlite3
import time
import tempfile
import os
import pytest
import multiprocessing
import threading

from ..job_manager.model_scheduler import Model_scheduler 


@pytest.fixture
def db_scheduler():
    """Create a temporary SQLite DB with the same schema for testing."""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    path = tmp.name
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE models (
            model_id TEXT PRIMARY KEY,
            norm TEXT,
            constraint_val REAL,
            adv_train INTEGER,
            grad_norm INTEGER DEFAULT 0,
            init_id INTEGER,
            current_epoch INTEGER DEFAULT 0,
            last_progress_ts INTEGER,
            last_time_selected INTEGER,
            status TEXT DEFAULT 'waiting',
            job_id TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Initialize your scheduler instance
    sch = Model_scheduler(db_path=path)
    yield sch

    os.remove(path)


def insert_model(sch, **fields):
    """Helper to insert a model record."""
    conn = sqlite3.connect(sch.db_path)
    c = conn.cursor()
    defaults = dict(
        model_id=None,
        norm="linf",
        constraint_val=8.0,
        adv_train=1,
        grad_norm=0,
        init_id=0,
        current_epoch=0,
        last_progress_ts=None,
        last_time_selected=None,
        status="waiting",
        job_id=None
    )
    defaults.update(fields)
    c.execute("""
        INSERT INTO models (model_id, norm, constraint_val, adv_train, grad_norm, init_id,
                            current_epoch, last_progress_ts, last_time_selected, status, job_id)
        VALUES (:model_id, :norm, :constraint_val, :adv_train, :grad_norm, :init_id,
                :current_epoch, :last_progress_ts, :last_time_selected, :status, :job_id)
    """, defaults)
    conn.commit()
    conn.close()


# ---- Tests ----

def test_update_progress_epoch_end_increments_epoch_and_updates_timestamp(db_scheduler):
    model_id = insert_model(db_scheduler, status="training", current_epoch=5)
    db_scheduler.update_progress_epoch_end(model_id)

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
    db_scheduler.update_progress_epoch_end(model_id)

    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT status FROM models WHERE model_id=?", (model_id,))
    (status,) = c.fetchone()
    conn.close()

    assert status == "finished"


def test_requeue_stale_trainings_sets_stale_training_to_waiting(db_scheduler):
    old_ts = int(time.time()) - 11 * 3600  # 11 hours ago
    model_id = insert_model(db_scheduler, status="training", last_progress_ts=old_ts, current_epoch=50)
    affected = db_scheduler.requeue_stale_trainings(threshold_hours=10)

    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT status FROM models WHERE model_id=?", (model_id,))
    (status,) = c.fetchone()
    conn.close()

    assert affected == 1
    assert status == "waiting"


def test_claim_next_waiting_model_sets_training_and_updates_last_time_selected(db_scheduler):
    # Insert a model with status 'waiting'
    model_id = insert_model(db_scheduler, status="waiting")

    # Claim the next waiting model
    claimed = db_scheduler.claim_next_waiting_model()

    # Debugging: Ensure the claimed model is not None
    assert claimed is not None, "No model was claimed. Check claim_next_waiting_model logic."

    # Verify the status and last_time_selected fields in the database
    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT status, last_time_selected FROM models WHERE model_id=?", (model_id,))
    row = c.fetchone()
    conn.close()

    # Debugging: Ensure the row exists
    assert row is not None, f"Model with model_id {model_id} not found in the database."

    status, last_time_selected = row

    # Assertions
    assert status == "training", f"Expected status 'training', got '{status}'"
    assert isinstance(last_time_selected, int), "last_time_selected should be an integer timestamp."
    
# ---- Concurrency / locking tests ----

def _claim_in_subprocess(db_path, return_dict, key):
    """Helper: run claim_next_waiting_model() in a subprocess."""
    sch = Model_scheduler(db_path)
    try:
        result = sch.claim_next_waiting_model()

        # Debugging: Print the result for inspection
        print(f"[DEBUG] Subprocess {key} claimed result: {result}")

        return_dict[key] = result
    except Exception as e:
        print(f"[ERROR] Subprocess {key} encountered an error: {e}")
        return_dict[key] = None


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

    # Debugging: Print results for inspection
    print("[DEBUG] Results from subprocesses:", results)

    # Ensure all succeeded
    successes = [r["result"] for r in results if r and r["result"]]
    assert len(successes) == 5, f"Expected 5 models claimed, got {len(successes)}"

    # All model_ids must be unique
    model_ids = [r["model_id"] for r in successes]

    # Debugging: Print claimed model IDs
    print("[DEBUG] Claimed model IDs:", model_ids)

    assert len(set(model_ids)) == 5, f"Duplicate model IDs claimed: {model_ids}"

    # Verify DB consistency
    conn = sqlite3.connect(db_scheduler.db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM models WHERE status='training'")
    (training_count,) = c.fetchone()
    conn.close()

    assert training_count == 5, f"Expected 5 models in 'training' status, got {training_count}"

def test_requeue_and_claim_do_not_deadlock(db_scheduler):
    """Ensure requeue_stale_trainings and claim_next_waiting_model can run concurrently without freezing."""

    # Insert one training and one waiting model
    m1 = insert_model(db_scheduler, status="training", current_epoch=5,
                      last_progress_ts=int(time.time()) - 11 * 3600)
    m2 = insert_model(db_scheduler, status="waiting", current_epoch=0)

    def run_requeue():
        db_scheduler.requeue_stale_trainings(threshold_hours=10)

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

def test_concurrent_claims_with_slurm_jobs(db_scheduler):
    """
    Test that multiple Slurm array jobs can claim models concurrently without conflicts.
    Ensure that each job gets a unique model_id.
    """
    # Insert 6 waiting models
    for i in range(6):
        insert_model(db_scheduler, model_id=f"model_{i+1}", status="waiting")

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []

    # Simulate 6 Slurm array jobs trying to claim models concurrently
    for i in range(6):
        p = multiprocessing.Process(
            target=_claim_in_subprocess,
            args=(db_scheduler.db_path, return_dict, f"job_{i+1}")
        )
        procs.append(p)
        p.start()

    # Wait for all processes to complete
    for p in procs:
        p.join()

    # Collect results
    claimed_model_ids = [result["model_id"] for result in return_dict.values() if result]

    # Ensure all claimed model_ids are unique
    assert len(claimed_model_ids) == 6, f"Expected 6 models claimed, got {len(claimed_model_ids)}"
    assert len(set(claimed_model_ids)) == 6, f"Duplicate model IDs claimed: {claimed_model_ids}"

    print("[PASS] All Slurm jobs claimed unique models.")