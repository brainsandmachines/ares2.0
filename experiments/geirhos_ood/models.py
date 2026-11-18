"""The model set: every HF-published checkpoint in the five contstim model JSONs, grouped into families.

The five files (``epsilon_bounded_contstim/model_zoo/models_{main62,dvd,swin,vit,left_out}.json``,
since 2026-11-17) hold every zoo run exactly once. The set keeps what Hugging Face publishes
(201 today) and drops the same two kinds of run ``model_store.hf_export`` keeps off the Hub:

- runs matched by ``model_store/hf_excludes.csv`` (collapsed / gradient-masked / overfit);
- the DVD ``_resetepoch`` continuations (``model_store/zoo_excludes.csv``; the contepoch
  variants are the published ones).

Families come from the JSON fields: ``<arch> madry_<norm>`` / ``<arch> trades_<norm>`` for the
standard runs, ``dvdb_`` / ``v1_`` prefixes for DVD-B and V1, ``gradnorm_<norm>`` for GradNorm,
and one ``clean baselines`` family. Order: clean baselines, then convnext_base, vit_b_cvst,
swin_b; within an arch standard, DVD-B, V1, GradNorm; madry before trades; l1, l2, linf; within
a family by (eps, init).

``generator`` drives the reference styling: solid bar + bold label for the adv_experiments
generators (``adv_campaign_role == "generator"``), light bar for everything else (their
black-box other inits, and models outside the campaign: V1, GradNorm, swin/vit TRADES).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from model_store.build_experiments import load_excludes
from model_store.hf_export import HF_EXCLUDES_CSV, excluded_reason

ZOO_DIR = Path.home() / "Documents/epsilon_bounded_contstim/model_zoo"
ZOO_FILES = tuple(ZOO_DIR / f"models_{tag}.json" for tag in ("main62", "dvd", "swin", "vit", "left_out"))
BASELINES_FAMILY = "clean baselines"

ARCHS = ("convnext_base", "vit_b_cvst", "swin_b")
NORMS = ("l1", "l2", "linf")
# protocol -> (family prefix, rank within an arch)
PROTOCOLS = {"madry_trades": ("", 0), "dvd_b": ("dvdb_", 1), "v1": ("v1_", 2), "gradnorm": ("gradnorm_", 3)}
BASELINE_PROTOCOLS = ("clean_baseline", "v1_baseline")
BASELINE_VARIANTS = ("standard", "DVD-B", "clean", "noise")  # vit/swin baselines carry their arch name


@dataclass(frozen=True)
class ModelEntry:
    key: str              # zoo name with '/' -> '_'
    arch: str
    kind: str             # checkpoint kind: best / last / advbest
    checkpoint: Path
    family: str
    generator: bool       # True = adv_experiments generator (solid bar), False = anything else
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


def zoo_entries(files: Iterable[Path] = ZOO_FILES) -> list[tuple[str, dict]]:
    """(file name, entry) for every entry of ``files``, in file order."""
    return [(Path(p).name, m) for p in files for m in json.loads(Path(p).read_text())["models"]]


def skip_reason(name: str, excludes: dict[str, str]) -> str | None:
    """Why ``name`` is not on the Hub, or None if it is published."""
    if "_resetepoch" in name:
        return "DVD _resetepoch continuation"
    return excluded_reason(name, excludes)


def family_of(m: dict) -> tuple[str, tuple]:
    """(family name, sort key of the family) from a JSON entry; raise if it fits none."""
    arch, protocol, norm = m["arch"], m["protocol"], m["norm"]
    if arch not in ARCHS:
        raise KeyError(f"{m['model_name']}: unknown arch {arch!r}")
    if protocol in BASELINE_PROTOCOLS:
        return BASELINES_FAMILY, (-1,)
    if protocol not in PROTOCOLS or norm is None:
        raise KeyError(f"{m['model_name']}: no family for protocol {protocol!r}, norm {norm!r}")
    prefix, rank = PROTOCOLS[protocol]
    trades = norm.endswith("trades")
    base = norm[:-len("trades")] if trades else norm
    if base not in NORMS:
        raise KeyError(f"{m['model_name']}: unknown norm {norm!r}")
    name = f"{prefix}{base}" if protocol == "gradnorm" else f"{prefix}{'trades' if trades else 'madry'}_{base}"
    return f"{arch} {name}", (ARCHS.index(arch), rank, trades, NORMS.index(base))


def _baseline_rank(m: dict) -> tuple:
    variant = m["variant"] if m["variant"] in BASELINE_VARIANTS else "standard"
    return (ARCHS.index(m["arch"]), BASELINE_PROTOCOLS.index(m["protocol"]),
            BASELINE_VARIANTS.index(variant), m["init"])


def load_models(files: Iterable[Path] = ZOO_FILES, excludes_csv: Path = HF_EXCLUDES_CSV) -> list[ModelEntry]:
    """The HF-published JSON checkpoints in plot order; raise if one cannot be placed in a family."""
    excludes = load_excludes(excludes_csv)
    keyed = []
    seen: set[str] = set()
    for source, m in zoo_entries(files):
        key = m["model_name"].replace("/", "_")
        if skip_reason(key, excludes):
            continue
        ckpt = m["checkpoint"]
        if ckpt in seen:
            raise ValueError(f"{ckpt} listed twice across the zoo JSONs")
        seen.add(ckpt)
        family, fam_rank = family_of(m)
        within = _baseline_rank(m) if family == BASELINES_FAMILY else (m["eps"], m["init"])
        keyed.append(((fam_rank, within), ModelEntry(
            key=key, arch=m["arch"], kind=m["checkpoint_kind"], checkpoint=Path(ckpt), family=family,
            generator=m.get("adv_campaign_role") == "generator", eps=m["eps"], init=m["init"],
            source_json=source,
        )))
    return [e for _k, e in sorted(keyed, key=lambda t: t[0])]


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
