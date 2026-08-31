"""Work out which (checkpoint kind, norm, eps) cells of the AutoAttack grid a model still needs.

Pure functions over already-read CSV text, so the same code runs against the local AIRCC sshfs
mount and, shipped over ssh, against the BGU cluster filesystem -- and is unit-testable without
either.

The row-matching rules deliberately mirror ``data_analysis/autoattack_array_eval.observed_settings``
(``model_name`` equality plus checkpoint *basename* matching). That basename fallback is what lets a
row written on AIRCC under a relative ``results/models/...`` path still count after the model has
been staged to an absolute path on the BGU cluster.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

Cell = tuple[str, float]


def grid_cells(norms, eps_inputs) -> set[Cell]:
    return {(str(norm).strip().lower(), round(float(eps), 10)) for norm in norms for eps in eps_inputs}


def observed_cells(csv_text: str, model_name: str, ckpt_filename: str) -> set[Cell]:
    """(norm, eps_input) pairs already recorded for this model+checkpoint in one sweep CSV."""
    found: set[Cell] = set()
    if not csv_text:
        return found
    for row in csv.DictReader(io.StringIO(csv_text)):
        if (row.get("model_name") or "").strip() != model_name:
            continue
        row_ckpt = (row.get("checkpoint_path") or "").strip()
        if not row_ckpt or row_ckpt.replace("\\", "/").rsplit("/", 1)[-1] != ckpt_filename:
            continue
        norm = (row.get("attack_norm") or "").strip().lower()
        eps = row.get("epsilon_input")
        if not norm or eps in (None, ""):
            continue
        try:
            found.add((norm, round(float(eps), 10)))
        except (TypeError, ValueError):
            continue
    return found


@dataclass
class KindStatus:
    """State of one checkpoint kind of one model, unified across both clusters and Botero."""

    kind: str
    ckpt_on_slurm: bool = False
    ckpt_on_aircc: bool = False
    ckpt_on_botero: bool = False
    missing: set[Cell] = field(default_factory=set)

    @property
    def has_checkpoint(self) -> bool:
        """A checkpoint the *cluster* sweep can reach, possibly after staging.

        Deliberately excludes Botero: ``runnable`` gates sbatch submission, and a checkpoint that
        exists only in the local archive is not something the cluster can attack. The Botero lane
        keys off ``ckpt_on_botero`` instead.
        """
        return self.ckpt_on_slurm or self.ckpt_on_aircc

    @property
    def needs_staging(self) -> bool:
        """The checkpoint has to be copied over before this kind can be swept."""
        return bool(self.missing) and not self.ckpt_on_slurm and self.ckpt_on_aircc

    @property
    def runnable(self) -> bool:
        """There is work to do and a checkpoint to do it with (possibly after staging)."""
        return bool(self.missing) and self.has_checkpoint


def kind_status(
    kind: str,
    ckpt_filename: str,
    model_name: str,
    grid: set[Cell],
    slurm_files: set[str],
    slurm_csv_text: str,
    aircc_files: set[str],
    aircc_csv_text: str,
    botero_files: set[str] = frozenset(),
    botero_csv_text: str = "",
) -> KindStatus:
    """Fold all three views of one checkpoint kind into a single status.

    Cells count as done if *any* side already has them:

    * BGU -- where the sbatch lane runs.
    * AIRCC -- staging carries its CSV across (merging it when the BGU side has one too), so an
      AIRCC-only row is a row the BGU job will find and skip.
    * Botero -- the local lane writes its results only into the local archive and never pushes them
      to either cluster, so without counting them here the nightly run would keep re-submitting
      cells Botero has already computed.

    The Botero arguments default to empty so every pre-existing caller keeps its old behaviour.
    """
    observed = (
        observed_cells(slurm_csv_text, model_name, ckpt_filename)
        | observed_cells(aircc_csv_text, model_name, ckpt_filename)
        | observed_cells(botero_csv_text, model_name, ckpt_filename)
    )
    return KindStatus(
        kind=kind,
        ckpt_on_slurm=ckpt_filename in slurm_files,
        ckpt_on_aircc=ckpt_filename in aircc_files,
        ckpt_on_botero=ckpt_filename in botero_files,
        missing=grid - observed,
    )
