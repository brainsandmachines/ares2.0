"""Offline tests for command construction: the batch divisor and the resume shift.

No GPU/Slurm/torch needed -- ``_peek_resume_epoch`` is stubbed wherever a real
checkpoint would otherwise have to be read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slurm_job_manager import lifecycle
from slurm_job_manager.db import JobDB


@pytest.fixture
def db(tmp_path):
    return JobDB(str(tmp_path / "jobs.sqlite"))


def _row(**over):
    row = {
        "model_name": "m", "init_mode": "scratch", "dependency_model_name": "",
        "resume_offset_assumed": "", "model": "convnext_base",
        "training.epochs": "200", "training.batch_size": "256",
    }
    row.update(over)
    return row


def _val(cmd, key):
    for t in cmd:
        if t.startswith(f"{key}="):
            return t.split("=", 1)[1]
    return None


def _count(cmd, key):
    return sum(1 for t in cmd if t.startswith(f"{key}="))


# --------------------------------------------------------------------------
# batch divisor (the bsz/2 rtx6000 rule)
# --------------------------------------------------------------------------
def test_batch_divisor_unset_is_noop(db, monkeypatch, tmp_path):
    monkeypatch.delenv("SJM_BATCH_DIVISOR", raising=False)
    cmd = lifecycle.build_command(_row(), tmp_path, db, python_exe="py")
    assert _val(cmd, "training.batch_size") == "256"


def test_batch_divisor_two_halves_the_csv_value(db, monkeypatch, tmp_path):
    monkeypatch.setenv("SJM_BATCH_DIVISOR", "2")
    cmd = lifecycle.build_command(_row(), tmp_path, db, python_exe="py")
    assert _val(cmd, "training.batch_size") == "128"


def test_batch_divisor_floors_at_one(db, monkeypatch, tmp_path):
    monkeypatch.setenv("SJM_BATCH_DIVISOR", "512")
    cmd = lifecycle.build_command(_row(training__batch_size=None), tmp_path, db, python_exe="py")
    assert _val(cmd, "training.batch_size") == "1"


def test_batch_divisor_applies_to_shift_managed_rows(db, monkeypatch, tmp_path):
    """The divisor must run over the FINAL token list, shifted tokens included."""
    monkeypatch.setenv("SJM_BATCH_DIVISOR", "2")
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 108)
    db.upsert_pending("dep", 150, 1)
    db.set_best_checkpoint("dep", "/ckpt/dep/model_best.pth.tar", 80.0)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="150")
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "training.batch_size") == "128"
    assert _val(cmd, "training.epochs") == "158"


# --------------------------------------------------------------------------
# resume shift
# --------------------------------------------------------------------------
def test_non_resume_row_is_not_shifted(db, tmp_path):
    """resume_offset_assumed is ignored unless init_mode is exactly 'resume'."""
    cmd = lifecycle.build_command(
        _row(init_mode="scratch", resume_offset_assumed="150"), tmp_path, db, python_exe="py")
    assert _val(cmd, "training.epochs") == "200"


def test_resume_without_offset_uses_static_csv_epochs(db, tmp_path):
    db.upsert_pending("dep", 150, 1)
    db.set_best_checkpoint("dep", "/ckpt/dep/model_best.pth.tar", 80.0)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="")
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "training.epochs") == "200"
    assert _count(cmd, "training.epochs") == 1


def test_shift_derives_target_from_actual_resume_epoch(db, monkeypatch, tmp_path):
    """The real gradnorm case: dep best is at epoch 107, not the assumed 150.

    target = start_epoch(108) + (csv 200 - offset 150) = 158, so the row trains
    the intended 50 new epochs rather than 92.
    """
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 108)
    db.upsert_pending("dep", 150, 1)
    db.set_best_checkpoint("dep", "/ckpt/dep/model_best.pth.tar", 80.0)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="150")
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "training.epochs") == "158"
    assert _count(cmd, "training.epochs") == 1  # never emitted twice


def test_shift_moves_epsilon_schedule_boundaries_too(db, monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 108)
    db.upsert_pending("dep", 150, 1)
    db.set_best_checkpoint("dep", "/ckpt/dep/model_best.pth.tar", 80.0)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="150",
               **{"epsilon_schedule.ramp_start_epoch": "150",
                  "epsilon_schedule.ramp_end_epoch": "160"})
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "epsilon_schedule.ramp_start_epoch") == "108"   # 150 - 42
    assert _val(cmd, "epsilon_schedule.ramp_end_epoch") == "118"     # 160 - 42


def test_zero_shift_reproduces_static_csv_values(db, monkeypatch, tmp_path):
    """dep really was at its assumed epoch -> the CSV values pass through."""
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 200)
    db.upsert_pending("dep", 200, 1)
    db.set_best_checkpoint("dep", "/ckpt/dep/last.pth.tar", 31.4)
    row = _row(init_mode="resume", dependency_model_name="dep",
               resume_offset_assumed="200", **{"training.epochs": "240"})
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "training.epochs") == "240"


def test_failed_peek_falls_back_to_static_csv(db, monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: None)
    db.upsert_pending("dep", 150, 1)
    db.set_best_checkpoint("dep", "/ckpt/dep/model_best.pth.tar", 80.0)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="150")
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "training.epochs") == "200"


def test_target_is_cached_and_survives_a_later_restart(db, monkeypatch, tmp_path):
    """A restart resumes from own_last (a LATER epoch) but must keep the target.

    Without the sidecar cache the target would be re-derived from the new,
    higher start epoch and the run would grow longer on every restart.
    """
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 108)
    db.upsert_pending("dep", 150, 1)
    db.set_best_checkpoint("dep", "/ckpt/dep/model_best.pth.tar", 80.0)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="150")
    lifecycle.build_command(row, tmp_path, db, python_exe="py")

    sidecar = tmp_path / "m" / ".aircc_resume_target.json"
    assert json.loads(sidecar.read_text())["target_epochs"] == 158

    # Now the model has its own checkpoint at a much later epoch.
    (tmp_path / "m" / "last.pth.tar").write_text("x")
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 150)
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "training.epochs") == "158"      # cached, not 200
    assert _val(cmd, "model.resume").endswith("m/last.pth.tar")


def test_migrated_aircc_sidecar_is_honoured(db, monkeypatch, tmp_path):
    """A sidecar copied over from AIRCC alongside last.pth.tar is reused as-is."""
    d = tmp_path / "m"
    d.mkdir()
    (d / "last.pth.tar").write_text("x")
    (d / ".aircc_resume_target.json").write_text(json.dumps(
        {"target_epochs": 149, "start_epoch": 99, "resume_offset_assumed": 150,
         "resume_path": "/shared/.../convnext_base_baseline_init0/model_best.pth.tar"}))
    # Would resolve to something else entirely if the cache were ignored.
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 500)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="150")
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "training.epochs") == "149"


def test_shift_syncs_total_epochs_into_the_db(db, monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 108)
    db.upsert_pending("m", 200, 10)
    db.upsert_pending("dep", 150, 1)
    db.set_best_checkpoint("dep", "/ckpt/dep/model_best.pth.tar", 80.0)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="150")
    lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert db.get("m").total_epochs == 158


def test_corrupt_sidecar_is_recomputed(db, monkeypatch, tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "last.pth.tar").write_text("x")
    (d / ".aircc_resume_target.json").write_text("{not json")
    monkeypatch.setattr(lifecycle, "_peek_resume_epoch", lambda p: 108)
    row = _row(init_mode="resume", dependency_model_name="dep", resume_offset_assumed="150")
    cmd = lifecycle.build_command(row, tmp_path, db, python_exe="py")
    assert _val(cmd, "training.epochs") == "158"
