"""Shared CSV schema + loading for the AIRCC job manager.

The CSVs in ``csv/`` are the single source of truth for hyperparameters and
dependencies. Each row has two kinds of columns:

* **Metadata columns** -- used by the manager's own logic (claiming, dependency
  resolution, best-checkpoint scoring). NOT emitted as Hydra overrides.
* **Override columns** -- the header IS the exact Hydra key; the manager emits
  ``<header>=<cell>`` for every NON-EMPTY cell (and nothing for empty cells).
  Dynamic checkpoint args (``continuation.checkpoint_path`` / ``model.resume``)
  and ``+machine=aircc`` are added at launch time, not stored here.

This module is imported by the generator, the manager, the seeder and the
dashboard so the column list is defined exactly once.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Iterable, Optional

CSV_DIR = Path(__file__).resolve().parent / "csv"
ARCHES = ("convnext_small", "convnext_base", "convnext_large")

# Columns the manager reads but never passes to training.
METADATA_COLUMNS = [
    "model_name", "arch", "init", "protocol", "init_mode", "epoch_variant",
    "dependency_model_name", "threat_norm", "threat_eps", "priority", "notes",
    # For init_mode="resume" rows only: the epoch the dependency's final
    # checkpoint was ASSUMED to be at when training.epochs / epsilon_schedule.*
    # were computed (e.g. 150 for gradnorm's baseline dep, or the source
    # model's target epoch count for dvd_b contepoch chains). At launch,
    # lifecycle.py peeks the ACTUAL resumed checkpoint's epoch and shifts
    # training.epochs + the 4 epsilon_schedule.* columns by the delta, so a
    # resume from an earlier (e.g. best-scoring, not final-epoch) checkpoint
    # still trains exactly the originally-intended number of NEW epochs.
    # Empty for every other row -- lifecycle.py leaves those untouched.
    "resume_offset_assumed",
]

# Columns whose header is the Hydra key; emitted as key=value when non-empty,
# in this order. Dynamic ckpt args + +machine=aircc are appended by the launcher.
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
    "lr_scheduler.lrb",
]

ALL_COLUMNS = METADATA_COLUMNS + OVERRIDE_COLUMNS


def build_overrides(row: dict, skip: Optional[Iterable[str]] = None) -> list[str]:
    """Return ``key=value`` Hydra tokens for every non-empty override cell.

    ``skip`` omits columns the caller will emit itself (e.g. lifecycle.py
    replaces training.epochs / epsilon_schedule.* with resume-shifted values
    for managed resume rows -- Hydra errors on a key passed twice).
    """
    skip_set = set(skip) if skip else set()
    out: list[str] = []
    for col in OVERRIDE_COLUMNS:
        if col in skip_set:
            continue
        val = str(row.get(col, "")).strip()
        if val != "":
            out.append(f"{col}={val}")
    return out


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


def load_spec(csv_dir: Path = CSV_DIR) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    """Load + VALIDATE the CSVs; return ``(rows, row_map, deps_map)``.

    The job manager calls this before every claim so CSV edits reach models that
    have not been claimed yet. Because ``generate_csvs.py`` rewrites the files in
    place (non-atomic ``DictWriter``), a reload can catch a half-written file --
    so this validates the whole snapshot and raises ``ValueError`` rather than
    handing back a partially-applied spec. The caller is expected to refuse to
    claim on error, never to fall back to an older snapshot.
    """
    rows = load_all_rows(csv_dir)
    if not rows:
        raise ValueError(f"no arch CSVs found in {csv_dir} (looked for {'/'.join(ARCHES)}.csv)")
    seen: set[str] = set()
    for i, r in enumerate(rows):
        name = (r.get("model_name") or "").strip()
        if not name:
            raise ValueError(f"{csv_dir}: row {i} has a blank model_name")
        if name in seen:
            raise ValueError(f"{csv_dir}: duplicate model_name {name!r}")
        seen.add(name)
        for col in ("priority", "training.epochs"):
            val = str(r.get(col, "") or "").strip()
            try:
                int(val)
            except ValueError:
                raise ValueError(f"{csv_dir}: {name}: {col}={val!r} is not an int") from None
    return rows, row_map(rows), deps_map(rows)
