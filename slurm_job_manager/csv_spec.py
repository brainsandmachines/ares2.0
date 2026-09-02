"""Shared CSV schema + loading for the Slurm job manager.

The CSVs in ``csv/`` are the single source of truth for hyperparameters and
dependencies. Each row has two kinds of columns:

* **Metadata columns** -- used by the manager's own logic (claiming, dependency
  resolution, test lane, best-checkpoint scoring). NOT emitted as Hydra
  overrides.
* **Override columns** -- the header IS the exact Hydra key; the manager emits
  ``<header>=<cell>`` for every NON-EMPTY cell. The dynamic checkpoint args
  (``continuation.checkpoint_path`` / ``model.resume``) are added at launch time.

This mirrors ``aircc/aircc_job_manager/csv_spec.py`` (same ``adversarial_training``
entrypoint, same override columns) with two Botero differences: an extra
``is_test`` metadata column for the priority lane, and **no** ``+machine`` tag
(Botero uses the config-default dataset dirs). ``training.batch_size`` holds the
full 96 GB (rtx_pro_6000) batch; the per-partition ``SJM_BATCH_DIVISOR`` env
halves it on rtx6000 (see ``lifecycle._apply_batch_divisor``).
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Iterable

CSV_DIR = Path(__file__).resolve().parent / "csv"
ARCHES = ("convnext_small", "convnext_base", "convnext_large", "vit-b-cvst_swin-b")

# Columns the manager reads but never passes to training.
METADATA_COLUMNS = [
    "model_name", "arch", "init", "protocol", "init_mode", "epoch_variant",
    "dependency_model_name", "threat_norm", "threat_eps", "priority", "is_test",
    "notes", "resume_offset_assumed",
]

# Columns whose header is the Hydra key; emitted as key=value when non-empty.
OVERRIDE_COLUMNS = [
    "model",
    "model.experiment_name",
    "model.experiment_num",
    "model.v1_noise_mode",
    "model.compile_model",
    "output_dir",
    "training.epochs",
    "training.batch_size",
    "attacks.advtrain",
    "attacks.attack_criterion",
    "attacks.attack_norm",
    "attacks.attack_domain",
    "attacks.attack_eps",
    "attacks.attack_it",
    "attacks.v1_attack_eps",
    "attacks.gradnorm",
    "attacks.gradnorm_penalty_norm",
    "dataset.dvd.enabled",
    "dataset.dvd.variant",
    "continuation.enabled",
    "continuation.use_ema",
    "checkpointing.save_best_adv",
    "epsilon_schedule.enabled",
    "epsilon_schedule.type",
    "epsilon_schedule.source_epsilon",
    "epsilon_schedule.target_epsilon",
    "epsilon_schedule.warmup_epochs",
    "epsilon_schedule.ramp_start_epoch",
    "epsilon_schedule.ramp_end_epoch",
    "epsilon_schedule.fixed_start_epoch",
    "dataset.mixup_active",
    "optimizer.weight_decay",
    "lr_scheduler.lrb",
    "lr_scheduler.warmup_epochs",
]

ALL_COLUMNS = METADATA_COLUMNS + OVERRIDE_COLUMNS


def build_overrides(row: dict, skip: Iterable[str] | None = None) -> list[str]:
    """Return ``key=value`` Hydra tokens for every non-empty override cell.

    ``skip`` omits columns the caller re-emits itself (Hydra errors on a
    duplicated key) -- used by the resume-shift path in ``lifecycle``.
    """
    skipped = set(skip or ())
    out: list[str] = []
    for col in OVERRIDE_COLUMNS:
        if col in skipped:
            continue
        val = str(row.get(col, "")).strip()
        if val != "":
            out.append(f"{col}={val}")
    return out


def is_test_row(row: dict) -> bool:
    return str(row.get("is_test", "")).strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_arch_rows(arch: str, csv_dir: Path = CSV_DIR) -> list[dict]:
    path = csv_dir / f"{arch}.csv"
    with path.open(newline="") as fh:
        return list(_csv.DictReader(fh))


def load_all_rows(csv_dir: Path = CSV_DIR) -> list[dict]:
    rows: list[dict] = []
    for arch in ARCHES:
        if (csv_dir / f"{arch}.csv").exists():
            rows.extend(load_arch_rows(arch, csv_dir))
    return rows


def row_map(rows: Iterable[dict]) -> dict[str, dict]:
    return {r["model_name"]: r for r in rows}


def deps_map(rows: Iterable[dict]) -> dict[str, str]:
    return {r["model_name"]: (r.get("dependency_model_name") or "").strip() for r in rows}
