"""Offline tests for the DB layer + stage transitions."""

from orchestrator.db import (
    OrchestratorDB,
    STAGE_TRAIN,
    STAGE_AA_SWEEP,
    STAGE_PLOT_RESULTS,
    STAGE_COMPLETED,
    next_stage,
)


def make_db(tmp_path):
    db = OrchestratorDB(str(tmp_path / "orch.db"))
    db.upsert_model("m1", "golan-trainmodels", "l2_8_init1", "/d/m1", 0, 0, 250)
    return db


def test_next_stage_pipeline():
    assert next_stage(STAGE_TRAIN) == STAGE_AA_SWEEP
    assert next_stage(STAGE_AA_SWEEP) == STAGE_PLOT_RESULTS
    assert next_stage(STAGE_PLOT_RESULTS) == STAGE_COMPLETED
    # terminal is idempotent
    assert next_stage(STAGE_COMPLETED) == STAGE_COMPLETED


def test_total_epochs(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("c", "eps_curriculum", "linf_cont4to8_ramp_init1", "/d/c", 3, 17, 15)
    assert db.get_model_state("c").total_epochs == 35


def test_epoch_update_and_stage_advance(tmp_path):
    db = make_db(tmp_path)
    db.update_epoch("m1", 7)
    assert db.get_model_state("m1").current_epoch == 7
    db.advance_stage("m1", STAGE_AA_SWEEP)
    assert db.get_model_state("m1").current_stage == STAGE_AA_SWEEP
    db.advance_stage("m1", STAGE_COMPLETED)
    row = db.get_model_state("m1")
    assert row.current_stage == STAGE_COMPLETED and row.status == "COMPLETED"


def test_upsert_preserves_runtime_state(tmp_path):
    db = make_db(tmp_path)
    db.mark_running("m1", 999, "rtx6000")
    db.update_epoch("m1", 42)
    # re-seed same row: runtime fields must survive
    db.upsert_model("m1", "golan-trainmodels", "l2_8_init1", "/d/m1", 0, 0, 250)
    row = db.get_model_state("m1")
    assert row.status == "RUNNING"
    assert row.current_epoch == 42
    assert row.slurm_job_id == 999


def test_next_pending_for_node_determinism(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    for mid in ["b", "a", "c"]:
        db.upsert_model(mid, "golan-trainmodels", mid, f"/d/{mid}", 0, 0, 250)
    # ordered by (priority, model_id) -> 'a' first
    assert db.next_pending_for_node("rtx6000").model_id == "a"


def test_node_pinning(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("pinned", "golan-trainmodels", "j", "/d/p", 0, 0, 250,
                    cluster_node="rtx_pro_6000")
    assert db.next_pending_for_node("rtx6000") is None
    assert db.next_pending_for_node("rtx_pro_6000").model_id == "pinned"


def test_requeue_stale_running(tmp_path):
    db = make_db(tmp_path)
    db.mark_running("m1", 1, "rtx6000")
    # a freshly-started job must NOT be requeued
    assert db.requeue_stale_running(3600) == 0
    # backdate progress, then it should be requeued
    db._write("UPDATE models_queue SET last_update_ts = 0 WHERE model_id = ?", ("m1",))
    assert db.requeue_stale_running(3600) == 1
    row = db.get_model_state("m1")
    assert row.status == "PENDING" and row.slurm_job_id is None


def test_failure_hash_store(tmp_path):
    db = make_db(tmp_path)
    assert not db.has_failure_hash("abc")
    db.record_failure_hash("abc", "m1", "/r/abc.md")
    assert db.has_failure_hash("abc")
    # idempotent insert
    db.record_failure_hash("abc", "m1", "/r/abc.md")
    assert db.has_failure_hash("abc")
