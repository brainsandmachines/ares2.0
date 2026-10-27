"""The selected-models set: 20 families (97 models) plus their clean baselines.

Each model resolves to its experiment checkpoint through the model zoo JSON
(``epsilon_bounded_contstim/model_zoo/models_for_adversarial_experiments.json``), which
names the exact checkpoint (kind + path) whose AutoAttack score is recorded. Always key
on that checkpoint path: the published name is not always the run dir (e.g.
``convnext_base_linftrades_cont4to8_init1`` is the ``..._init3`` run dir).

Each family also carries the other init (0 or 1) of every model when that run exists, at
the checkpoint kind ``models_aa_clean_by_init.csv`` picks for it, flagged ``in_json=False``.
A ``publish_as`` row in ``model_store/blessing_overrides.csv`` swaps in the run blessed under
that name (e.g. ``convnext_base_linftrades_cont4to8_init0`` is the ``..._init2`` rerun). Only
other inits listed in ``models_for_blackbox_experiments.json`` are kept: it leaves out the
failed (collapsed / catastrophically overfit) runs.
The clean baselines get the same treatment. Only convnext_base has a second init;
vit_b_cvst / swin_b have init1 only.

Used by ``shape_bias_selected.py`` (scoring) and ``plot_selected_models_report.py`` (PDF).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

ZOO_DIR = Path.home() / "Documents/epsilon_bounded_contstim/model_zoo"
ZOO_JSON = ZOO_DIR / "models_for_adversarial_experiments.json"
BLACKBOX_JSON = ZOO_DIR / "models_for_blackbox_experiments.json"
ZOO_BY_INIT_CSV = ZOO_DIR / "models_aa_clean_by_init.csv"
MODELS_ROOT = Path("/mnt/data4t/models")
BLESSING_OVERRIDES_CSV = Path(__file__).resolve().parents[1] / "model_store" / "blessing_overrides.csv"
COMPARED_INITS = (0, 1)

_CONVNEXT = {
    "madry_l1": "l1_1_init0 l1_2_init0 l1_4_init0 l1_cont4to6_init0 l1_cont4to8_init0",
    "madry_l2": "l2_1_init0 l2_2_init0 l2_4_init0 l2_cont4to6_init0 l2_cont4to8_init1",
    "madry_linf": "linf_1_init0 linf_2_init0 linf_4_init0 linf_cont4to6_init0 linf_cont4to8_init0",
    "trades_l1": "l1trades_1_init0 l1trades_2_init0 l1trades_4_init0 l1trades_cont4to6_init0 l1trades_cont4to8_init0",
    "trades_l2": "l2trades_1_init0 l2trades_2_init0 l2trades_4_init0 l2trades_cont4to6_init0 l2trades_cont4to8_init0",
    "trades_linf": "linftrades_1_init0 linftrades_2_init0 linftrades_4_init1 linftrades_cont4to6_init0 "
                   "linftrades_cont4to8_init1",
    "dvdb_madry_l1": "dvd_b_l1_1_init0 dvd_b_l1_2_init0 dvd_b_l1_4_init1 dvd_b_l1_cont4to6_init1_contepoch "
                     "dvd_b_l1_cont4to8_init1_contepoch",
    "dvdb_madry_l2": "dvd_b_l2_1_init0 dvd_b_l2_2_init1 dvd_b_l2_4_init0 dvd_b_l2_cont4to6_init0_contepoch "
                     "dvd_b_l2_cont4to8_init0_contepoch",
    "dvdb_madry_linf": "dvd_b_linf_1_init0 dvd_b_linf_2_init1 dvd_b_linf_4_init0 dvd_b_linf_cont4to6_init0_contepoch "
                       "dvd_b_linf_cont4to8_init0_contepoch",
    "dvdb_trades_l1": "dvd_b_l1trades_1_init0 dvd_b_l1trades_2_init0 dvd_b_l1trades_4_init0 "
                      "dvd_b_l1trades_cont4to6_init0_contepoch dvd_b_l1trades_cont4to8_init0_contepoch",
    "dvdb_trades_l2": "dvd_b_l2trades_1_init0 dvd_b_l2trades_2_init0 dvd_b_l2trades_4_init0 "
                      "dvd_b_l2trades_cont4to6_init0_contepoch dvd_b_l2trades_cont4to8_init0_contepoch",
    "dvdb_trades_linf": "dvd_b_linftrades_1_init0 dvd_b_linftrades_2_init1 dvd_b_linftrades_4_init0 "
                        "dvd_b_linftrades_cont4to6_init0_contepoch dvd_b_linftrades_cont4to8_init0_contepoch",
    "v1_madry_l2": "v1_l2_1_init0 v1_l2_2_init0 v1_l2_4_init0 v1_l2_cont4to6_init0 v1_l2_cont4to8_init0",
    "gradnorm_l1": "gradnorm_l1_1_init0 gradnorm_l1_2_init0",
}
_MADRY_INIT1 = "{n}_1_init1 {n}_2_init1 {n}_4_init1 {n}_cont4to6_init1 {n}_cont4to8_init1"

# The clean (non-adversarially-trained) baselines: shape bias only, no AA sweep.
BASELINES = [
    "convnext_base_baseline_init0", "convnext_base_dvd_b_baseline_init0", "convnext_base_v1_clean_init0",
    "convnext_base_v1_noise_init0", "vit_b_cvst_baseline_init1", "swin_b_baseline_init1",
]

# family name -> (arch, baseline key, [zoo keys in x-axis order])
FAMILIES: dict[str, tuple[str, str, list[str]]] = {}
for _fam, _names in _CONVNEXT.items():
    _baseline = ("convnext_base_dvd_b_baseline_init0" if _fam.startswith("dvdb")
                 else "convnext_base_v1_clean_init0" if _fam.startswith("v1")
                 else "convnext_base_baseline_init0")
    FAMILIES[f"convnext_base {_fam}"] = ("convnext_base", _baseline, [f"convnext_base_{n}" for n in _names.split()])
for _arch in ("vit_b_cvst", "swin_b"):
    for _norm in ("l1", "l2", "linf"):
        FAMILIES[f"{_arch} madry_{_norm}"] = (
            _arch, f"{_arch}_baseline_init1",
            [f"{_arch}_{n}" for n in _MADRY_INIT1.format(n=_norm).split()],
        )


@dataclass(frozen=True)
class Selected:
    key: str              # zoo name with '/' -> '_' (e.g. vit_b_cvst_l1_1_init1)
    arch: str
    kind: str             # best / last / advbest
    checkpoint: Path
    eps: float | None     # training eps (6/8 for the cont4to6/cont4to8 runs)
    init: int             # published init (the init3 rerun published as init1 counts as 1)
    in_json: bool = True  # the zoo JSON's pick for this (arch, protocol, norm, eps)

    @property
    def run_dir(self) -> str:
        return self.checkpoint.parent.name


def load_zoo(json_path: Path = ZOO_JSON) -> dict[str, Selected]:
    models = json.loads(Path(json_path).read_text())["models"]
    return {
        m["model_name"].replace("/", "_"): Selected(
            key=m["model_name"].replace("/", "_"),
            arch=m["arch"],
            kind=m["checkpoint_kind"],
            checkpoint=Path(m["checkpoint"]),
            eps=m["eps"],
            init=m["init"],
        )
        for m in models
    }


def published_as_overrides() -> dict[str, tuple[str, str | None]]:
    """{published name: (run dir, checkpoint kind or None)} from the blessing overrides."""
    from data_analysis.autoattack_array_eval import CHECKPOINT_KIND_CANDIDATES

    kind_of = {f: kind for kind, files in CHECKPOINT_KIND_CANDIDATES.items() for f in files}
    df = pd.read_csv(BLESSING_OVERRIDES_CSV).dropna(subset=["publish_as"])
    return {r.publish_as: (r.model, None if pd.isna(r.checkpoint) else kind_of[r.checkpoint])
            for r in df.itertuples()}


def other_inits(sel: Selected, by_init: pd.DataFrame) -> list[Selected]:
    """The same (arch, protocol, norm, eps) at the other compared init(s), if those runs exist."""
    from data_analysis.autoattack_array_eval import find_checkpoint_for_kind

    overrides = published_as_overrides()

    init_cols = [c for c in by_init.columns if c.startswith("init") and c.endswith("_model")]
    rows = by_init[by_init[init_cols].eq(sel.key.replace(f"{sel.arch}_", f"{sel.arch}/", 1)).any(axis=1)
                   | by_init[init_cols].eq(sel.key).any(axis=1)]
    if len(rows) != 1:
        raise KeyError(f"{sel.key}: expected one row in {ZOO_BY_INIT_CSV.name}, found {len(rows)}")
    row = rows.iloc[0]
    others = []
    for init in COMPARED_INITS:
        name, kind = row[f"init{init}_model"], row[f"init{init}_ckpt"]
        if init == sel.init or pd.isna(name) or row[f"init{init}_status"] != "finished":
            continue
        run_name = name.replace("/", "_")
        if name in overrides:
            run_name, kind = overrides[name][0], overrides[name][1] or kind
        run_dir = MODELS_ROOT / sel.arch / run_name
        checkpoint = find_checkpoint_for_kind(run_dir, kind)
        if checkpoint is None:
            raise FileNotFoundError(f"No '{kind}' checkpoint in {run_dir}")
        others.append(replace(sel, key=name.replace("/", "_"), kind=kind, checkpoint=checkpoint,
                              init=init, in_json=False))
    return others


def resolve(json_path: Path = ZOO_JSON) -> tuple[dict[str, list[Selected]], dict[str, Selected]]:
    """Return ({family: [Selected, ...]}, {baseline key: Selected}); raise if any name is missing.

    Each family list is ordered by (eps, init) and holds the JSON pick plus the other init.
    """
    zoo = load_zoo(json_path)
    wanted = [*BASELINES, *(k for _a, _b, ks in FAMILIES.values() for k in ks)]
    missing = sorted({k for k in wanted if k not in zoo})
    if missing:
        raise KeyError(f"Not in {json_path}: {', '.join(missing)}")
    by_init = pd.read_csv(ZOO_BY_INIT_CSV)
    blackbox = load_zoo(BLACKBOX_JSON)

    def kept_other_inits(sel: Selected) -> list[Selected]:
        kept = [m for m in other_inits(sel, by_init) if m.key in blackbox]
        for m in kept:
            if (m.checkpoint, m.kind) != (blackbox[m.key].checkpoint, blackbox[m.key].kind):
                raise ValueError(f"{m.key}: {m.kind} {m.checkpoint} here, but {BLACKBOX_JSON.name} has "
                                 f"{blackbox[m.key].kind} {blackbox[m.key].checkpoint}")
        return kept

    families = {
        fam: sorted((m for k in keys for m in [zoo[k], *kept_other_inits(zoo[k])]),
                    key=lambda m: (m.eps, m.init))
        for fam, (_a, _b, keys) in FAMILIES.items()
    }
    baselines = {m.key: m for b in BASELINES for m in [zoo[b], *kept_other_inits(zoo[b])]}
    return families, baselines
