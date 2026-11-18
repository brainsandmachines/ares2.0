"""The model set: every checkpoint in the two HF-zoo JSONs, grouped like the shape-bias report.

``models_for_adversarial_experiments.json`` (the experiment picks, 133) plus
``models_for_blackbox_experiments.json`` (their other init, 66) -> 199 checkpoints. Each
entry's family and its position come from ``data_analysis.selected_models.resolve()`` (the
set behind the last page of ``selected_models_aa_shape_bias.pdf``), so the bars line up
with the shape-bias plot. The vit_b_cvst / swin_b TRADES models are not in that report;
they get ``<arch> trades_<norm>`` families after the arch's madry families, ordered by
(eps, init) like the rest.

``in_adv_json`` drives the reference styling: solid bar + bold label for the experiment
pick, light bar for the other init.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from data_analysis.selected_models import ZOO_DIR, resolve

ADV_JSON = ZOO_DIR / "models_for_adversarial_experiments.json"
BLACKBOX_JSON = ZOO_DIR / "models_for_blackbox_experiments.json"
BASELINES_FAMILY = "clean baselines"


@dataclass(frozen=True)
class ModelEntry:
    key: str              # zoo name with '/' -> '_' (the shape-bias report's label stem)
    arch: str
    kind: str             # checkpoint kind: best / last / advbest
    checkpoint: Path
    family: str
    in_adv_json: bool     # True = experiment pick (solid bar), False = other init
    eps: float | None
    init: int
    source_json: str

    @property
    def run_dir(self) -> str:
        return self.checkpoint.parent.name

    @property
    def model_id(self) -> str:
        """Stable id for result files: the run dir plus the checkpoint kind."""
        return f"{self.run_dir}__{self.kind}"

    @property
    def label(self) -> str:
        return f"{self.key} [{self.kind}]"


def _json_models(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["models"]


def load_models(adv_json: Path = ADV_JSON, blackbox_json: Path = BLACKBOX_JSON,
                reference: tuple[dict, dict] | None = None) -> list[ModelEntry]:
    """All JSON checkpoints in plot order; raise if one cannot be placed in a family.

    ``reference`` is ``resolve()``'s ({family: [Selected]}, {baseline: Selected}); tests pass a
    fixture instead of reading the zoo.
    """
    families, baselines = reference if reference is not None else resolve()
    ordered = {BASELINES_FAMILY: list(baselines.values()), **families}
    position = {str(s.checkpoint): (fam, i) for i, (fam, sels) in enumerate(ordered.items())
                for s in sels}
    rank = {str(s.checkpoint): j for sels in ordered.values() for j, s in enumerate(sels)}
    family_order = list(ordered)

    entries: list[ModelEntry] = []
    seen: set[str] = set()
    for path, in_adv in ((adv_json, True), (blackbox_json, False)):
        for m in _json_models(path):
            ckpt = m["checkpoint"]
            if ckpt in seen:
                raise ValueError(f"{ckpt} listed twice across the zoo JSONs")
            seen.add(ckpt)
            if ckpt in position:
                family = position[ckpt][0]
            elif in_adv and m["arch"] in ("vit_b_cvst", "swin_b") and str(m["norm"]).endswith("trades"):
                family = f"{m['arch']} trades_{m['norm'][:-len('trades')]}"
            else:
                raise KeyError(f"{m['model_name']} ({ckpt}): no family in the shape-bias reference set")
            if family not in family_order:
                madry = f"{m['arch']} madry_"
                last_madry = max(i for i, f in enumerate(family_order) if f.startswith(madry))
                # after the arch's madry block, keeping trades_l1 < trades_l2 < trades_linf
                after = [i for i, f in enumerate(family_order)
                         if f.startswith(f"{m['arch']} trades_") and f < family]
                family_order.insert(max([last_madry, *after]) + 1, family)
            entries.append(ModelEntry(
                key=m["model_name"].replace("/", "_"), arch=m["arch"], kind=m["checkpoint_kind"],
                checkpoint=Path(ckpt), family=family, in_adv_json=in_adv, eps=m["eps"],
                init=m["init"], source_json=Path(path).name,
            ))

    def sort_key(e: ModelEntry):
        ck = str(e.checkpoint)
        within = rank[ck] if ck in rank else (e.eps if e.eps is not None else -1, e.init)
        return family_order.index(e.family), (0, within) if ck in rank else (1, within)

    return sorted(entries, key=sort_key)


def group_by_family(entries: list[ModelEntry]) -> dict[str, list[ModelEntry]]:
    groups: dict[str, list[ModelEntry]] = {}
    for e in entries:
        groups.setdefault(e.family, []).append(e)
    return groups


def select_models(entries: list[ModelEntry], names: list[str] | None) -> list[ModelEntry]:
    """Filter by key, run dir or model_id; unknown names raise."""
    if not names:
        return entries
    picked = [e for e in entries if {e.key, e.run_dir, e.model_id} & set(names)]
    known = {n for e in picked for n in (e.key, e.run_dir, e.model_id)}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise KeyError(f"Unknown model(s): {unknown}")
    return picked
