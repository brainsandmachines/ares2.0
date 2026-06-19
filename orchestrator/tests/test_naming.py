"""Tests for arch/protocol/eps derivation and the DB migration."""

import sqlite3

from orchestrator import naming
from orchestrator.db import OrchestratorDB


def test_arch_from_name():
    assert naming.arch_from_name("l2_8_init1") == "convnext_small"
    assert naming.arch_from_name("convnext_base_l2_8_init1") == "convnext_base"
    assert naming.arch_from_name("convnext_large_linf_4_init2") == "convnext_large"


def test_protocol_from_name():
    assert naming.protocol_from_name("l2_8_init1") == "madry"
    assert naming.protocol_from_name("linf_8_init1") == "madry"
    assert naming.protocol_from_name("l2trades_8_init1") == "trades"
    assert naming.protocol_from_name("gradnorm_l2_8_init1") == "gradnorm"
    assert naming.protocol_from_name("dvd_l2_8_init1") == "dvd"
    assert naming.protocol_from_name("v1clean_l2_8_init1") == "v1"
    assert naming.protocol_from_name("baseline_init1") == "baseline"


def test_eps_from_name():
    assert naming.eps_from_name("l2_8_init1") == 8.0
    assert naming.eps_from_name("linf_16_init2") == 16.0
    assert naming.eps_from_name("baseline_init1") is None
    # curriculum: target epsilon
    assert naming.eps_from_name("linf_cont4to8_ramp_init1") == 8.0
    assert naming.eps_from_name("convnext_base_l2_2_init1") == 2.0


def test_model_dir_for():
    root = "/r/models"
    assert naming.model_dir_for("l2_8_init1", root) == "/r/models/convnext_small_l2_8_init1"
    assert naming.model_dir_for("convnext_base_l2_8_init1", root) == "/r/models/convnext_base_l2_8_init1"


def test_upsert_stores_new_columns(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("convnext_base_l2_8_init1", "golan-trainmodels",
                    "convnext_base_l2_8_init1", "/d", 0, 0, 250,
                    arch="convnext_base", protocol="madry", eps=8.0)
    row = db.get_model_state("convnext_base_l2_8_init1")
    assert row.arch == "convnext_base" and row.protocol == "madry" and row.eps == 8.0


def test_migration_adds_columns_to_legacy_db(tmp_path):
    """An old DB without arch/protocol/eps gets them via init_db migration."""
    p = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(p)
    conn.execute(
        """
        CREATE TABLE models_queue (
            model_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'PENDING',
            current_stage TEXT NOT NULL DEFAULT 'TRAIN', cluster_node TEXT,
            slurm_job_id INTEGER, launcher TEXT, job_name TEXT, model_dir TEXT,
            current_epoch INTEGER NOT NULL DEFAULT 0, warmup_epochs INTEGER NOT NULL DEFAULT 0,
            linear_epochs INTEGER NOT NULL DEFAULT 0, plateau_epochs INTEGER NOT NULL DEFAULT 0,
            last_error_hash TEXT, priority INTEGER NOT NULL DEFAULT 100, last_update_ts INTEGER
        )
        """
    )
    conn.execute("INSERT INTO models_queue (model_id, plateau_epochs) VALUES ('old', 250)")
    conn.commit()
    conn.close()

    db = OrchestratorDB(p)  # triggers migration
    row = db.get_model_state("old")
    assert row.arch == "convnext_small"   # NOT NULL default backfilled
    assert row.protocol is None and row.eps is None
