"""Offline tests for the eval-lane job top-up (squeue/sbatch stubbed)."""

from __future__ import annotations

import subprocess

import pytest

from slurm_job_manager import eval_lane
from slurm_job_manager.db import JobDB


@pytest.fixture
def db(tmp_path):
    return JobDB(str(tmp_path / "jobs.sqlite"))


def _eval_row(db, name, owner):
    db.upsert_pending(name, 10, 1)
    db.claim_next(owner, {})
    db.update_epoch(name, 10)
    db.mark_trained(name)


class FakeSlurm:
    def __init__(self, pending_jobs=0, squeue_rc=0, sbatch_rc=0):
        self.pending_jobs, self.squeue_rc, self.sbatch_rc = pending_jobs, squeue_rc, sbatch_rc
        self.sbatched = 0

    def __call__(self, argv, **kw):
        if argv[0] == "squeue":
            assert "-n" in argv and eval_lane.EVAL_JOB_NAME in argv and "PENDING" in argv
            out = "".join(f"{100 + i}\n" for i in range(self.pending_jobs))
            return subprocess.CompletedProcess(argv, self.squeue_rc, out, "")
        assert argv[0] == "sbatch" and argv[-1] == str(eval_lane.EVAL_SBATCH)
        if self.sbatch_rc:
            return subprocess.CompletedProcess(argv, self.sbatch_rc, "", "denied")
        self.sbatched += 1
        return subprocess.CompletedProcess(argv, 0, f"{5000 + self.sbatched}\n", "")


def test_submits_one_job_per_unowned_eval_row_minus_queued(db):
    for i in range(3):
        _eval_row(db, f"m{i}", owner=i + 1)
    slurm = FakeSlurm(pending_jobs=1)
    r = eval_lane.top_up(db, run=slurm)
    assert r["unowned"] == 3 and r["pending_jobs"] == 1
    assert r["submitted"] == ["5001", "5002"] and slurm.sbatched == 2


def test_nothing_to_do_never_calls_slurm(db):
    def run(argv, **kw):
        raise AssertionError("no slurm call expected")
    assert eval_lane.top_up(db, run=run)["submitted"] == []


def test_squeue_failure_submits_nothing(db):
    _eval_row(db, "a", 1)
    slurm = FakeSlurm(squeue_rc=1)
    r = eval_lane.top_up(db, run=slurm)
    assert r["pending_jobs"] is None and r["submitted"] == [] and slurm.sbatched == 0


def test_sbatch_failure_is_reported_not_raised(db):
    _eval_row(db, "a", 1)
    r = eval_lane.top_up(db, run=FakeSlurm(sbatch_rc=1))
    assert r["submitted"] == []


def test_promotes_stray_trained_pending_rows_and_dry_run_writes_nothing(db):
    _eval_row(db, "a", 1)
    db.claim_next(7, {}, lane="eval")
    db.release("a", owner=7)                 # back to eval, unowned
    db.upsert_pending("stray", 10, 1)
    db.claim_next(8, {})
    db.update_epoch("stray", 10)
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    conn.execute("UPDATE jobs SET status='pending', slurm_job_id=NULL WHERE model_name='stray'")
    conn.commit()
    conn.close()

    slurm = FakeSlurm()
    r = eval_lane.top_up(db, dry_run=True, run=slurm)
    assert r["promoted"] == ["stray"] and r["unowned"] == 2
    assert r["submitted"] == ["(dry-run)"] * 2 and slurm.sbatched == 0
    assert db.get("stray").status == "pending"

    r = eval_lane.top_up(db, run=slurm)
    assert r["promoted"] == ["stray"] and db.get("stray").status == "eval"
    assert slurm.sbatched == 2
