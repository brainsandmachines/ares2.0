"""Unit tests for the resume-shift fix: init_mode="resume" rows whose dep's
DB-best checkpoint isn't its final epoch must still train exactly
(csv training.epochs - resume_offset_assumed) NEW epochs -- see lifecycle.py's
_resolve_resume_target / _shifted_resume_overrides / RESUME_SHIFT_COLUMNS.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aircc.aircc_job_manager import lifecycle
from aircc.aircc_job_manager.csv_spec import ALL_COLUMNS
from aircc.aircc_job_manager.db import AirccDB


def _base_row(**overrides) -> dict:
    row = {c: "" for c in ALL_COLUMNS}
    row.update(
        model_name="convnext_base_gradnorm_l1_1_init0",
        arch="convnext_base",
        init="0",
        protocol="gradnorm",
        init_mode="resume",
        dependency_model_name="convnext_base_baseline_init0",
        **{"training.epochs": "200", "training.batch_size": "256"},
        resume_offset_assumed="150",
    )
    row.update(overrides)
    return row


def _seed_dep(db: AirccDB, dep_ckpt: Path) -> None:
    db.upsert_pending("convnext_base_baseline_init0", 150, 0)
    db.mark_finished("convnext_base_baseline_init0")
    db.set_best_checkpoint("convnext_base_baseline_init0", str(dep_ckpt), 81.5)


def _write_ckpt(path: Path, *, epoch: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "state_dict": {}}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# --------------------------------------------------------------------------
def test_first_launch_shifts_to_actual_peeked_epoch(tmp_path):
    """dep's best checkpoint is mid-training (epoch 101, not the final 150) --
    training.epochs must become 101+50=151, not the CSV's static 200."""
    models_root = tmp_path / "models"
    dep_ckpt = tmp_path / "dep" / "model_best.pth.tar"
    _write_ckpt(dep_ckpt, epoch=101)

    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    _seed_dep(db, dep_ckpt)
    db.upsert_pending("convnext_base_gradnorm_l1_1_init0", 200, 1)

    cmd = lifecycle.build_command(_base_row(), models_root, db, python_exe="python3")

    assert f"model.resume={dep_ckpt}" in cmd
    assert "training.epochs=151" in cmd
    assert not any(c.startswith("training.epochs=200") for c in cmd)
    # exactly one training.epochs override on the command line (no Hydra dup-key)
    assert sum(c.startswith("training.epochs=") for c in cmd) == 1


def test_restart_reuses_persisted_target_not_current_own_last_epoch(tmp_path):
    """Second launch (own_last now exists, further along) must reuse the SAME
    target computed on first launch, not re-derive a smaller one from
    own_last's now-later epoch (which would shrink remaining training)."""
    models_root = tmp_path / "models"
    dep_ckpt = tmp_path / "dep" / "model_best.pth.tar"
    _write_ckpt(dep_ckpt, epoch=101)

    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    _seed_dep(db, dep_ckpt)
    db.upsert_pending("convnext_base_gradnorm_l1_1_init0", 200, 1)

    first_cmd = lifecycle.build_command(_base_row(), models_root, db, python_exe="python3")
    assert "training.epochs=151" in first_cmd

    # simulate progress: own_last now exists, far past the original start
    own_last = lifecycle._own_last(models_root, "convnext_base_gradnorm_l1_1_init0")
    _write_ckpt(own_last, epoch=140)

    second_cmd = lifecycle.build_command(_base_row(), models_root, db, python_exe="python3")
    assert f"model.resume={own_last}" in second_cmd
    assert "training.epochs=151" in second_cmd  # unchanged, not 140+50=190


def test_epsilon_schedule_columns_shift_uniformly_for_dvd_contepoch(tmp_path):
    row = _base_row(
        protocol="madry",
        dependency_model_name="convnext_base_dvd_b_l1_4_init0",
        resume_offset_assumed="200",
    )
    row.update({
        "training.epochs": "240",
        "epsilon_schedule.enabled": "true",
        "epsilon_schedule.warmup_epochs": "204",
        "epsilon_schedule.ramp_start_epoch": "205",
        "epsilon_schedule.ramp_end_epoch": "225",
        "epsilon_schedule.fixed_start_epoch": "226",
    })

    models_root = tmp_path / "models"
    dep_ckpt = tmp_path / "dep" / "model_best.pth.tar"
    _write_ckpt(dep_ckpt, epoch=190)  # 10 epochs earlier than the assumed offset (200)

    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    db.upsert_pending("convnext_base_dvd_b_l1_4_init0", 200, 0)
    db.mark_finished("convnext_base_dvd_b_l1_4_init0")
    db.set_best_checkpoint("convnext_base_dvd_b_l1_4_init0", str(dep_ckpt), 60.0)
    db.upsert_pending(row["model_name"], 240, 1)

    cmd = lifecycle.build_command(row, models_root, db, python_exe="python3")

    # shift = 190 - 200 = -10, applied uniformly to all 5 columns
    assert "training.epochs=230" in cmd
    assert "epsilon_schedule.warmup_epochs=194" in cmd
    assert "epsilon_schedule.ramp_start_epoch=195" in cmd
    assert "epsilon_schedule.ramp_end_epoch=215" in cmd
    assert "epsilon_schedule.fixed_start_epoch=216" in cmd


def test_peek_failure_falls_back_to_static_csv_values(tmp_path):
    """Dep checkpoint has no 'epoch' key (e.g. weights-only) -> can't shift ->
    must fall back to the original static values, not crash or drop the key."""
    models_root = tmp_path / "models"
    dep_ckpt = tmp_path / "dep" / "model_best.pth.tar"
    dep_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}}, dep_ckpt)  # no 'epoch' key

    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    _seed_dep(db, dep_ckpt)
    db.upsert_pending("convnext_base_gradnorm_l1_1_init0", 200, 1)

    cmd = lifecycle.build_command(_base_row(), models_root, db, python_exe="python3")
    assert "training.epochs=200" in cmd


def test_unmanaged_resume_row_is_completely_unaffected(tmp_path):
    """Rows without resume_offset_assumed (i.e. every resume row before this
    fix, and any not yet regenerated) must behave exactly as before."""
    models_root = tmp_path / "models"
    dep_ckpt = tmp_path / "dep" / "model_best.pth.tar"
    _write_ckpt(dep_ckpt, epoch=101)

    db = AirccDB(str(tmp_path / "jobs.sqlite"))
    _seed_dep(db, dep_ckpt)
    db.upsert_pending("convnext_base_gradnorm_l1_1_init0", 200, 1)

    row = _base_row(resume_offset_assumed="")
    cmd = lifecycle.build_command(row, models_root, db, python_exe="python3")
    assert "training.epochs=200" in cmd
    target_file = lifecycle._resume_target_file(models_root, row["model_name"])
    assert not target_file.exists()
