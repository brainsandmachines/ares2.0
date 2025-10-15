import sqlite3
import time
import tempfile
import os
import pytest

from adv_scheduler import TaskScheduler  # replace with actual class name


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
