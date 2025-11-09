import sqlite3
import multiprocessing
import time
import shutil
import tempfile
import os
from model_scheduler import Model_scheduler


def insert_model(scheduler, status="waiting", norm="l2", constraint_val=1):
    model_id = scheduler._make_model_id(norm, constraint_val, True, init_id=None, grad_norm=0)
    scheduler._execute_sqlite(
        "INSERT OR IGNORE INTO models (model_id, norm, constraint_val, adv_train, grad_norm, init_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (model_id, norm, constraint_val, 1, 0, None, status),
        quiet_duplicate=True,
    )
    return model_id


def _claim_in_subprocess(db_path, return_dict, name):
    try:
        s = Model_scheduler(db_path=db_path)
        claimed = s.claim_next_waiting_model()
        return_dict[name] = {
            "result": bool(claimed),
            "model_id": claimed["model_id"] if claimed else None,
        }
    except Exception as e:
        return_dict[name] = {"result": False, "error": str(e)}


def test_concurrent_multi_claims_no_lock_or_duplicate(original_db="model_scheduler.db"):
    # --- make a temp copy of the original DB ---
    tmpdir = tempfile.mkdtemp()
    test_db_path = os.path.join(tmpdir, "model_scheduler_test.db")
    shutil.copy2(original_db, test_db_path)
    print(f"[INFO] Running test on copy: {test_db_path}")

    scheduler = Model_scheduler(db_path=test_db_path)

    # Clean test data (so we don’t mess up existing entries)
    conn = sqlite3.connect(test_db_path)
    conn.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    # Insert 5 waiting models
    for i in range(5):
        insert_model(scheduler, status="waiting", norm="l2", constraint_val=i + 1)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []

    for i in range(5):
        p = multiprocessing.Process(
            target=_claim_in_subprocess,
            args=(test_db_path, return_dict, f"p{i+1}"),
        )
        procs.append(p)
        p.start()

    # Wait for all processes (timeout safety)
    for p in procs:
        p.join(timeout=15)
        assert not p.is_alive(), "One of the processes hung (possible DB lock)."

    results = list(return_dict.values())
    print("[DEBUG] Results from subprocesses:", results)

    # Count successes
    successes = [r for r in results if r and r["result"]]
    print(f"[INFO] {len(successes)} models claimed successfully")

    model_ids = [r["model_id"] for r in successes if r["model_id"]]
    print("[DEBUG] Claimed model IDs:", model_ids)
    assert len(set(model_ids)) == len(model_ids), f"Duplicate IDs detected: {model_ids}"

    # Verify DB consistency
    conn = sqlite3.connect(test_db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM models WHERE status='training'")
    (training_count,) = c.fetchone()
    conn.close()
    print(f"[INFO] Models in 'training' status: {training_count}")

    assert training_count == len(successes), f"Expected {len(successes)} models in training, got {training_count}"
    print("[✅] Concurrency test passed successfully!")

    # Clean up temp dir after success
    shutil.rmtree(tmpdir)
    print("[INFO] Temporary test database removed.")


if __name__ == "__main__":
    test_concurrent_multi_claims_no_lock_or_duplicate()
