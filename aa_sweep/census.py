"""Work out which (checkpoint kind, norm, eps) cells of the AutoAttack grid a model still needs.

Pure functions over already-read CSV text, so the same code runs against a local model dir and,
shipped over ssh, against the BGU cluster filesystem -- and is unit-testable without either.

**One lane at a time.** A status describes exactly one machine's view of one checkpoint kind: the
files that machine has and the CSV that machine wrote. It deliberately does *not* union in what
another lane has computed, because the thing that ultimately decides which cells get attacked is
the engine on that machine diffing its own CSV against the grid. Counting a row that lives only on
the other machine would make this planner skip a cell that the machine in question will never
actually compute.

That is safe only because the two lanes own disjoint model sets (see plan.build_plan): a model with
a directory on the BGU cluster is the cluster's, everything else finished on AIRCC is Botero's. If
that split is ever loosened, this is the assumption that has to be revisited.

The row-matching rules deliberately mirror ``data_analysis/autoattack_array_eval.observed_settings``
(``model_name`` equality plus checkpoint *basename* matching). That basename fallback is what lets a
row written under a relative ``results/models/...`` path still count against a checkpoint now read
from an absolute path somewhere else.
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
    """One machine's state for one checkpoint kind of one model."""

    kind: str
    has_checkpoint: bool = False
    missing: set[Cell] = field(default_factory=set)

    @property
    def runnable(self) -> bool:
        """There is work to do, and a checkpoint on *this* machine to do it with."""
        return bool(self.missing) and self.has_checkpoint


def kind_status(
    kind: str,
    ckpt_filename: str,
    model_name: str,
    grid: set[Cell],
    files: set[str],
    csv_text: str,
) -> KindStatus:
    """Status of one checkpoint kind from one machine's own files and its own sweep CSV."""
    return KindStatus(
        kind=kind,
        has_checkpoint=ckpt_filename in files,
        missing=grid - observed_cells(csv_text, model_name, ckpt_filename),
    )
