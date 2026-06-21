"""Controller orchestration tests (stage execution + slurm liveness injected)."""

from orchestrator import controller
from orchestrator.db import (
    OrchestratorDB,
    STATUS_AA_EVAL,
    STATUS_FINISHED,
    STATUS_PLOTTING,
    STATUS_TRAINING,
)


def make_db(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("m1", "golan-trainmodels", "l2_8_init1", "/d/m1", 0, 0, 4)
    return db


def _advancing_runner(db, fail_at=None, fail_log=""):
    """Fake run_stage that mimics the in-job hooks by advancing DB status.

    ``fail_at`` (a stage name) makes that stage 'fail' (return rc=1 and not
    advance), with ``fail_log`` as the captured tail.
    """
    advance = {"train": STATUS_AA_EVAL, "aa": STATUS_PLOTTING, "plot": STATUS_FINISHED}

    def run_stage(stage, job):
        if stage == fail_at:
            return 1, fail_log
        # success: the real hooks would set this; emulate it here.
        if stage == "plot":
            db.mark_finished(job.model_id, "best")
        else:
            db.set_status(job.model_id, advance[stage])
        return 0, "ok"

    return run_stage


def test_full_pipeline_from_training(tmp_path):
    db = make_db(tmp_path)
    job = db.claim_next("rtx6000", 1)          # PENDING -> TRAINING
    assert job.status == STATUS_TRAINING
    rc = controller.run_job(db, job, run_stage=_advancing_runner(db))
    assert rc == 0
    assert db.get_model_state("m1").status == STATUS_FINISHED


def test_pipeline_entry_at_aa(tmp_path):
    db = make_db(tmp_path)
    db.force_status("m1", STATUS_AA_EVAL)
    job = db.claim_next("rtx6000", 1)          # stays AA_EVAL
    assert job.status == STATUS_AA_EVAL
    calls = []
    base = _advancing_runner(db)

    def tracking(stage, job):
        calls.append(stage)
        return base(stage, job)

    rc = controller.run_job(db, job, run_stage=tracking)
    assert rc == 0 and calls == ["aa", "plot"]      # training skipped
    assert db.get_model_state("m1").status == STATUS_FINISHED


def test_pipeline_entry_at_plotting(tmp_path):
    db = make_db(tmp_path)
    db.force_status("m1", STATUS_PLOTTING)
    job = db.claim_next("rtx6000", 1)
    calls = []
    base = _advancing_runner(db)
    rc = controller.run_job(db, job, run_stage=lambda s, j: (calls.append(s), base(s, j))[1])
    assert rc == 0 and calls == ["plot"]
    assert db.get_model_state("m1").status == STATUS_FINISHED


def test_transient_failure_requeues(tmp_path):
    db = make_db(tmp_path)
    job = db.claim_next("rtx6000", 55)
    runner = _advancing_runner(db, fail_at="train", fail_log="CANCELLED DUE TO TIME LIMIT")
    rc = controller.run_job(db, job, run_stage=runner)
    assert rc == 1
    row = db.get_model_state("m1")
    assert row.status == STATUS_TRAINING        # status preserved
    assert row.slurm_job_id is None and row.requeued == 1   # released for retry


def test_deterministic_error_marks_failed(tmp_path):
    db = make_db(tmp_path)
    job = db.claim_next("rtx6000", 7)
    runner = _advancing_runner(db, fail_at="train",
                               fail_log="ModuleNotFoundError: No module named 'dvd'")
    rc = controller.run_job(db, job, run_stage=runner)
    assert rc == 1
    assert db.get_model_state("m1").status == "FAILED"


def test_stage_success_but_no_status_advance_is_failure(tmp_path):
    db = make_db(tmp_path)
    job = db.claim_next("rtx6000", 7)

    def no_advance(stage, job):
        return 0, "ModuleNotFoundError: boom"   # rc 0 but status not advanced

    rc = controller.run_job(db, job, run_stage=no_advance)
    assert rc == 1
    assert db.get_model_state("m1").status == "FAILED"


def test_requeue_stale_only_dead_jobs(tmp_path, monkeypatch):
    db = make_db(tmp_path)
    db.claim_next("rtx6000", 321)               # TRAINING, owned by 321
    db._write("UPDATE models_queue SET last_update_ts = 0 WHERE model_id = 'm1'")

    # alive -> not requeued
    n = controller.requeue_stale(db, slurm_active=lambda jid: True)
    assert n == 0 and db.get_model_state("m1").slurm_job_id == 321

    # dead -> requeued
    n = controller.requeue_stale(db, slurm_active=lambda jid: False)
    assert n == 1
    row = db.get_model_state("m1")
    assert row.slurm_job_id is None and row.requeued == 1
