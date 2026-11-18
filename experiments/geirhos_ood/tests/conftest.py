"""Synthetic fixtures: tiny stimuli trees and human CSVs. None of this is a real result."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from experiments.geirhos_ood.tasks import TASKS, TaskSpec

CATS = ["airplane", "bear", "knife", "oven"]


def image_names(task: TaskSpec, n_per_cond: int = 2) -> list[tuple[str, str, str]]:
    """(filename, condition, category) in upstream's naming scheme."""
    conds = task.data_conditions or ("0",)
    out, k = [], 0
    for cond in conds:
        for j in range(n_per_cond):
            cat = CATS[(k + j) % len(CATS)]
            k += 1
            if task.contains_sessions:
                out.append((f"{k:04d}_syn_s01_{cond}_{cat}_10_n0{k:07d}_{j}.png", cond, cat))
            else:
                out.append((f"{k:04d}_syn_s01_0_{cat}_00_{cat}{k}.png", "NaN", cat))
    return out


def build_root(root: Path, task_ids, n_subjects: int = 2, with_humans: bool = True) -> Path:
    for tid in task_ids:
        task = TASKS[tid]
        rows = image_names(task)
        for name, _cond, cat in rows:
            d = root / "datasets" / tid / ("dnn/session-1" if task.contains_sessions else cat)
            d.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (k := len(name) * 7 % 255, 0, 255 - k)).save(d / name)
        if with_humans:
            hd = root / "raw-data" / tid
            hd.mkdir(parents=True, exist_ok=True)
            for s in range(1, n_subjects + 1):
                recs = []
                for i, (name, cond, cat) in enumerate(rows):
                    resp = cat if (i + s) % 3 else CATS[(CATS.index(cat) + 1) % len(CATS)]
                    recs.append({"subj": f"subject-{s:02d}", "Session": 1, "trial": i + 1, "rt": 0.8,
                                 "object_response": resp, "category": cat,
                                 "condition": "0" if cond == "NaN" else cond, "imagename": name})
                pd.DataFrame(recs).to_csv(hd / f"{tid}_subject-{s:02d}_session_1.csv", index=False)
    return root


@dataclass(frozen=True)
class FakeSelected:
    key: str
    arch: str
    kind: str
    checkpoint: Path
    eps: float | None
    init: int
    in_json: bool = True


def zoo_entry(arch, name, kind, ckpt, norm=None, eps=None, init=0):
    return {"arch": arch, "protocol": "madry_trades", "variant": None, "norm": norm, "eps": eps, "init": init,
            "model_name": f"{arch}/{name}", "checkpoint_kind": kind, "checkpoint": str(ckpt)}


@pytest.fixture
def tiny_zoo(tmp_path):
    """A 5-checkpoint zoo: baseline + other init, one madry family + other init, one vit trades model."""
    ck = tmp_path / "models"

    def ckpt(arch, run, f="model_best.pth.tar"):
        p = ck / arch / run / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    b0 = ckpt("convnext_base", "convnext_base_baseline_init0")
    b1 = ckpt("convnext_base", "convnext_base_baseline_init1")
    m0 = ckpt("convnext_base", "convnext_base_l2_1_init0")
    m1 = ckpt("convnext_base", "convnext_base_l2_1_init1", "last.pth.tar")
    vm = ckpt("vit_b_cvst", "vit_b_cvst_l2_1_init1")
    vt = ckpt("vit_b_cvst", "vit_b_cvst_l2trades_1_init1")
    adv = [zoo_entry("convnext_base", "baseline_init0", "best", b0),
           zoo_entry("convnext_base", "l2_1_init0", "best", m0, "l2", 1, 0),
           zoo_entry("vit_b_cvst", "l2_1_init1", "best", vm, "l2", 1, 1),
           zoo_entry("vit_b_cvst", "l2trades_1_init1", "best", vt, "l2trades", 1, 1)]
    bb = [zoo_entry("convnext_base", "baseline_init1", "best", b1, init=1),
          zoo_entry("convnext_base", "l2_1_init1", "last", m1, "l2", 1, 1)]
    (tmp_path / "adv.json").write_text(json.dumps({"models": adv}))
    (tmp_path / "bb.json").write_text(json.dumps({"models": bb}))
    S = FakeSelected
    reference = (
        {"convnext_base madry_l2": [S("convnext_base_l2_1_init0", "convnext_base", "best", m0, 1, 0),
                                    S("convnext_base_l2_1_init1", "convnext_base", "last", m1, 1, 1, False)],
         "vit_b_cvst madry_l2": [S("vit_b_cvst_l2_1_init1", "vit_b_cvst", "best", vm, 1, 1)]},
        {"convnext_base_baseline_init0": S("convnext_base_baseline_init0", "convnext_base", "best", b0, None, 0),
         "convnext_base_baseline_init1": S("convnext_base_baseline_init1", "convnext_base", "best", b1, None, 1,
                                           False)},
    )
    return tmp_path / "adv.json", tmp_path / "bb.json", reference


@pytest.fixture
def tiny_models(tiny_zoo):
    from experiments.geirhos_ood.models import load_models

    adv, bb, ref = tiny_zoo
    return load_models(adv, bb, ref)
