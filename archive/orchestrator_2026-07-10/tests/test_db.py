"""Offline tests for the DB layer: schema, atomic claim, priority, requeue."""

import multiprocessing as mp
import time

from orchestrator.db import (
    OrchestratorDB,
    STATUS_AA_EVAL,
    STATUS_FINISHED,
    STATUS_PENDING,
    STATUS_PLOTTING,
    STATUS_TRAINING,
)


def make_db(tmp_path):
    db = OrchestratorDB(str(tmp_path / "orch.db"))
    db.upsert_model("m1", "golan-trainmodels", "l2_8_init1", "/d/m1", 0, 0, 250)
    return db


def test_upsert_sets_final_epoch_from_schedule(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("c", "eps_curriculum", "linf_cont4to8_ramp_init1", "/d/c", 3, 17, 15)
    row = db.get_model_state("c")
    assert row.total_epochs == 35
    assert row.target_epoch == 35  # final_epoch defaulted to the schedule sum


def test_final_epoch_override(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("c", "golan-trainmodels", "l2_8_init1", "/d/c", 0, 0, 150, final_epoch=42)
    assert db.get_model_state("c").target_epoch == 42


def test_epoch_update_writes_next_epoch(tmp_path):
    db = make_db(tmp_path)
    db.update_epoch("m1", 7)
    row = db.get_model_state("m1")
    assert row.current_epoch == 7 and row.next_epoch == 7


def test_claim_pending_becomes_training(tmp_path):
    db = make_db(tmp_path)
    claimed = db.claim_next("rtx6000", 111)
    assert claimed.model_id == "m1"
    assert claimed.status == STATUS_TRAINING       # PENDING -> TRAINING on claim
    assert claimed.slurm_job_id == 111
    # now owned -> not claimable again
    assert db.claim_next("rtx6000", 222) is None


def test_claim_priority_order(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    # all unowned/claimable, different statuses; PLOTTING should win.
    db.upsert_model("train", "golan-trainmodels", "l2_8_init1", "/d/t", 0, 0, 250)
    db.upsert_model("aa", "golan-trainmodels", "l2_8_init2", "/d/a", 0, 0, 250)
    db.upsert_model("plot", "golan-trainmodels", "l2_8_init3", "/d/p", 0, 0, 250)
    db.force_status("aa", STATUS_AA_EVAL)
    db.force_status("plot", STATUS_PLOTTING)
    assert db.claim_next("rtx6000", 1).model_id == "plot"
    assert db.claim_next("rtx6000", 2).model_id == "aa"
    assert db.claim_next("rtx6000", 3).model_id == "train"
    assert db.claim_next("rtx6000", 4) is None


def test_claim_protocol_priority_within_status(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("lo", "golan-trainmodels", "l2_8_init1", "/d/lo", 0, 0, 250, priority=100)
    db.upsert_model("hi", "golan-trainmodels", "l2_8_init2", "/d/hi", 0, 0, 250, priority=500)
    assert db.claim_next("rtx6000", 1).model_id == "hi"   # higher priority first


def test_claim_node_pinning(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("pinned", "golan-trainmodels", "j", "/d/p", 0, 0, 250,
                    cluster_node="rtx_pro_6000")
    assert db.claim_next("rtx6000", 1) is None
    assert db.claim_next("rtx_pro_6000", 2).model_id == "pinned"


def test_claimable_count(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    for i in range(3):
        db.upsert_model(f"m{i}", "golan-trainmodels", f"j{i}", f"/d/{i}", 0, 0, 250)
    assert db.claimable_count("rtx6000") == 3
    db.claim_next("rtx6000", 1)
    assert db.claimable_count("rtx6000") == 2


def test_requeue_releases_owner_and_bumps_counter(tmp_path):
    db = make_db(tmp_path)
    db.claim_next("rtx6000", 99)
    db.requeue("m1")
    row = db.get_model_state("m1")
    assert row.slurm_job_id is None and row.requeued == 1
    assert row.status == STATUS_TRAINING        # status preserved (resumes at stage)
    assert db.claim_next("rtx6000", 100).model_id == "m1"  # claimable again


def test_find_stale_candidates_by_threshold(tmp_path):
    db = make_db(tmp_path)
    db.claim_next("rtx6000", 7)  # TRAINING, owned, fresh
    thr = {STATUS_TRAINING: 3600, STATUS_AA_EVAL: 3600, STATUS_PLOTTING: 3600}
    assert db.find_stale_candidates(thr) == []          # fresh -> not stale
    db._write("UPDATE models_queue SET last_update_ts = 0 WHERE model_id = ?", ("m1",))
    stale = db.find_stale_candidates(thr)
    assert [r.model_id for r in stale] == ["m1"]


def test_get_by_slurm_job_id(tmp_path):
    db = make_db(tmp_path)
    db.claim_next("rtx6000", 4242)
    assert db.get_by_slurm_job_id(4242).model_id == "m1"
    assert db.get_by_slurm_job_id(9999) is None


def test_mark_finished_sets_status_best_ckpt_and_score(tmp_path):
    db = make_db(tmp_path)
    db.claim_next("rtx6000", 5)
    db.mark_finished("m1", "advbest", 57.3)
    row = db.get_model_state("m1")
    assert row.status == STATUS_FINISHED
    assert row.best_checkpoint == "advbest"
    assert row.best_score == 57.3
    assert row.slurm_job_id is None


def test_mark_failed_releases_owner(tmp_path):
    db = make_db(tmp_path)
    db.claim_next("rtx6000", 5)
    db.mark_failed("m1", "deadbeef")
    row = db.get_model_state("m1")
    assert row.status == "FAILED" and row.slurm_job_id is None
    assert row.last_error_hash == "deadbeef"


# --------------------------------------------------------------------------
# Atomic-claim concurrency: many processes hammer one shared sqlite file and
# must never double-claim a row.
# --------------------------------------------------------------------------
def _claim_worker(args):
    db_path, partition, slurm_id = args
    db = OrchestratorDB(db_path)
    row = db.claim_next(partition, slurm_id)
    return row.model_id if row else None


def test_atomic_claim_no_double_claim(tmp_path):
    db_path = str(tmp_path / "race.db")
    db = OrchestratorDB(db_path)
    n_models, n_workers = 20, 32
    for i in range(n_models):
        db.upsert_model(f"m{i:02d}", "golan-trainmodels", f"j{i}", f"/d/{i}", 0, 0, 250)

    args = [(db_path, "rtx6000", 10_000 + i) for i in range(n_workers)]
    with mp.Pool(n_workers) as pool:
        claimed = pool.map(_claim_worker, args)

    got = [c for c in claimed if c is not None]
    assert len(got) == n_models               # exactly the 20 rows claimed
    assert len(set(got)) == n_models          # each exactly once (no double-claim)
    assert claimed.count(None) == n_workers - n_models  # losers got nothing
    # DB agrees: every row owned by exactly one slurm id.
    owners = [db.get_model_state(f"m{i:02d}").slurm_job_id for i in range(n_models)]
    assert all(o is not None for o in owners)
    assert len(set(owners)) == n_models


def test_pending_dependency_gate(tmp_path):
    """A fresh PENDING cont row isn't claimable until its source is FINISHED."""
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("src", "golan-trainmodels", "l2_4_init1", "/d/src", 0, 0, 150)
    db.upsert_model("cont", "golan-trainmodels", "l2_cont4to8_ramp_init1", "/d/cont",
                    4, 26, 10, depends_on_model="src")
    # src is claimable; cont is gated on src not being FINISHED.
    assert db.claim_next("rtx6000", 1).model_id == "src"
    assert db.claim_next("rtx6000", 2) is None          # cont still gated
    db.mark_finished("src", "best", 50.0)               # source done -> dep satisfied
    assert db.claim_next("rtx6000", 3).model_id == "cont"


def test_resuming_cont_ignores_dependency_gate(tmp_path):
    """A resuming (non-PENDING) cont row is claimable even if the source isn't done."""
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("cont", "golan-trainmodels", "l2_cont4to8_ramp_init1", "/d/cont",
                    4, 26, 10, depends_on_model="src_not_in_db")
    db.force_status("cont", STATUS_TRAINING)            # simulate a requeued resume
    assert db.claim_next("rtx6000", 1).model_id == "cont"


def test_claim_rank_overrides_status_order(tmp_path):
    """Ranked rows are claimed first, in rank order, ahead of unranked rows even
    of higher status; once the list is exhausted, normal order resumes."""
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("aa", "golan-trainmodels", "l2_8_init1", "/d/aa", 0, 0, 250)
    db.force_status("aa", STATUS_AA_EVAL)                      # normally beats PENDING
    db.upsert_model("r2", "golan-trainmodels", "l2_8_init2", "/d/r2", 0, 0, 250)
    db.upsert_model("r1", "golan-trainmodels", "l2_8_init3", "/d/r1", 0, 0, 250)
    db.set_claim_rank("r1", 1)
    db.set_claim_rank("r2", 2)
    assert db.claim_next("rtx6000", 1).model_id == "r1"       # rank 1 first
    assert db.claim_next("rtx6000", 2).model_id == "r2"       # rank 2 next
    assert db.claim_next("rtx6000", 3).model_id == "aa"       # list done -> status order
    db.set_claim_rank("r1", None)                             # clearing works
    assert db.get_model_state("r1").claim_rank is None


def test_current_stage_tracks_status(tmp_path):
    """The legacy current_stage column stays coherent with the pipeline status
    through claim, the stage handoffs, and terminal completion."""
    db = make_db(tmp_path)
    db.claim_next("rtx6000", 1)                       # PENDING -> TRAINING
    assert db.get_model_state("m1").current_stage == "TRAIN"
    db.set_status("m1", STATUS_AA_EVAL)
    assert db.get_model_state("m1").current_stage == "AA_SWEEP"
    db.set_status("m1", STATUS_PLOTTING)
    assert db.get_model_state("m1").current_stage == "PLOT_RESULTS"
    db.mark_finished("m1", "best", 42.0)
    row = db.get_model_state("m1")
    assert row.status == STATUS_FINISHED and row.current_stage == "COMPLETED"


def test_mark_failed_leaves_stage(tmp_path):
    """FAILED has no pipeline stage, so current_stage is left untouched."""
    db = make_db(tmp_path)
    db.claim_next("rtx6000", 1)
    db.set_status("m1", STATUS_AA_EVAL)
    db.mark_failed("m1", "deadbeef")
    assert db.get_model_state("m1").current_stage == "AA_SWEEP"


def test_legacy_status_migration(tmp_path):
    """Pre-rebuild rows upgrade in place: status by stage, owner cleared on
    non-terminal rows, current_epoch carried into next_epoch."""
    db_path = str(tmp_path / "legacy.db")
    db = OrchestratorDB(db_path)
    for mid in ("a", "b", "c", "d", "e"):
        db.upsert_model(mid, "golan-trainmodels", f"j_{mid}", f"/d/{mid}", 0, 0, 250)
    # Simulate legacy rows directly (the 92 PENDING+AA_SWEEP case is the real DB).
    db._write("UPDATE models_queue SET status='PENDING', current_stage='AA_SWEEP', current_epoch=199 WHERE model_id='a'")
    db._write("UPDATE models_queue SET status='RUNNING', current_stage='TRAIN', current_epoch=50, slurm_job_id=999 WHERE model_id='b'")
    db._write("UPDATE models_queue SET status='COMPLETED', current_stage='COMPLETED' WHERE model_id='c'")
    db._write("UPDATE models_queue SET status='PENDING', current_stage='PLOT_RESULTS', current_epoch=199 WHERE model_id='d'")
    db._write("UPDATE models_queue SET status='PENDING', current_stage='TRAIN', current_epoch=108 WHERE model_id='e'")
    # Re-open -> init_db runs the idempotent migration.
    db2 = OrchestratorDB(db_path)
    a = db2.get_model_state("a")
    assert a.status == STATUS_AA_EVAL and a.slurm_job_id is None and a.next_epoch == 199
    b = db2.get_model_state("b")
    assert b.status == STATUS_TRAINING and b.slurm_job_id is None and b.next_epoch == 50
    assert db2.get_model_state("c").status == STATUS_FINISHED
    assert db2.get_model_state("d").status == STATUS_PLOTTING
    e = db2.get_model_state("e")
    assert e.status == STATUS_PENDING and e.next_epoch == 108   # PENDING+TRAIN unchanged
    # Idempotent: a second open changes nothing.
    db3 = OrchestratorDB(db_path)
    assert db3.get_model_state("a").status == STATUS_AA_EVAL
    assert db3.get_model_state("e").status == STATUS_PENDING
