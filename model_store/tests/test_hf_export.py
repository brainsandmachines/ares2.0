"""The Hugging Face export: EMA weights only, incremental, and never deleting.

These pin the properties the published repo depends on: the exported tensors are
``state_dict_ema`` and not ``state_dict``; an unchanged source is skipped while a
replaced one (new inode) is re-exported; a checkpoint with no EMA fails loudly and
blocks the upload instead of being published from ``state_dict``; orphans are
reported and left in place; a dry run writes nothing.

Checkpoints are tiny real ``torch.save`` files, so the torch/safetensors path runs.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from model_store import hf_export

COLUMNS = ["model", "relpath", "arch", "protocol", "norm", "kind", "rule",
           "db_source", "best_score", "note", "target"]


def _ckpt(path: Path, *, ema: bool = True, seed: int = 0, epoch: int = 7) -> dict:
    g = torch.Generator().manual_seed(seed)
    raw = {"stem.0.weight": torch.randn(4, 3, generator=g), "head.bias": torch.randn(4, generator=g)}
    ema_sd = {k: v + 100.0 for k, v in raw.items()}
    payload = {
        "epoch": epoch,
        "arch": "convnext_base",
        "state_dict": raw,
        "args": {"model": "convnext_base", "config": {
            "model": {"model": "convnext_base", "num_classes": 4},
            "attacks": {"advtrain": True, "attack_norm": "l2", "attack_eps": 4.0},
            "dataset": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                        "input_size": 224, "crop_pct": 0.875, "interpolation": "bicubic"},
            "training": {"model_ema_decay": 0.9998},
        }},
    }
    if ema:
        payload["state_dict_ema"] = ema_sd
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # a new inode every time, like backfill's rsync
    return ema_sd


def _row(tmp_path: Path, name: str, **ckpt_kw) -> tuple[dict, dict]:
    target = tmp_path / "models" / "convnext_base" / name / "last.pth.tar"
    ema = _ckpt(target, **ckpt_kw)
    return {"model": name, "relpath": f"convnext_base/madry/l2/{name}.pth.tar",
            "arch": "convnext_base", "protocol": "madry", "norm": "l2", "kind": "last",
            "rule": "db", "db_source": "sjm", "best_score": "51.5", "note": "",
            "target": str(target)}, ema


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "zoo" / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def _args(tmp_path: Path, manifest: Path, *extra: str) -> list[str]:
    return ["--manifest", str(manifest), "--mirror", str(tmp_path / "mirror"),
            "--hash-cache", str(tmp_path / "sha.jsonl"), *extra]


@pytest.fixture
def uploads(monkeypatch):
    calls = []
    monkeypatch.setattr(hf_export, "upload", lambda mirror, repo, nw: calls.append((mirror, repo)) or 0)
    return calls


def test_exports_ema_weights_not_state_dict(tmp_path, uploads):
    row, ema = _row(tmp_path, "m_l2_4_init0")
    manifest = _manifest(tmp_path, [row])

    assert hf_export.main(_args(tmp_path, manifest, "--apply", "--upload")) == 0

    dest = tmp_path / "mirror" / "convnext_base/madry/l2/m_l2_4_init0"
    exported = load_file(str(dest / "model.safetensors"))
    assert exported.keys() == ema.keys()
    for k in ema:
        assert torch.equal(exported[k], ema[k])
    meta = hf_export._read_meta(dest)
    assert meta["source"]["checkpoint"] == "last.pth.tar"
    assert meta["training"]["attack"]["attack_eps"] == 4.0
    assert meta["threat_model"] == {"norm": "l2", "eps_name": 4.0, "eps_internal": 4.0,
                                    "domain": "pixel"}
    manifest_row = (tmp_path / "mirror" / "manifest.csv").read_text().splitlines()
    assert manifest_row[0].split(",")[6:8] == ["eps_name", "eps_internal"]
    assert manifest_row[1].split(",")[5:8] == ["l2", "4.0", "4.0"]
    assert meta["preprocessing"]["input_size"] == 224
    assert meta["weights"]["parameters"] == 16
    assert uploads == [(tmp_path / "mirror", hf_export.HF_REPO)]
    assert not list(dest.glob(hf_export.TMP_PREFIX + "*"))
    assert "m_l2_4_init0" in (tmp_path / "mirror" / "README.md").read_text()
    assert "m_l2_4_init0" in (tmp_path / "mirror" / "manifest.csv").read_text()


def test_unchanged_source_is_skipped_and_a_replaced_one_reexported(tmp_path, uploads):
    row, _ = _row(tmp_path, "m_l2_4_init0")
    manifest = _manifest(tmp_path, [row])
    assert hf_export.main(_args(tmp_path, manifest, "--apply")) == 0

    mirror = tmp_path / "mirror"
    assert [i.status for i in hf_export.plan([row], mirror)] == ["unchanged"]

    new_ema = _ckpt(Path(row["target"]), seed=1, epoch=9)
    [item] = hf_export.plan([row], mirror)
    assert item.status == "changed"

    assert hf_export.main(_args(tmp_path, manifest, "--apply")) == 0
    exported = load_file(str(item.dest / "model.safetensors"))
    assert torch.equal(exported["head.bias"], new_ema["head.bias"])
    assert hf_export._read_meta(item.dest)["epoch"] == 9


def test_force_reexports_an_unchanged_source(tmp_path, uploads):
    row, _ = _row(tmp_path, "m_l2_4_init0")
    assert hf_export.main(_args(tmp_path, _manifest(tmp_path, [row]), "--apply")) == 0

    [item] = hf_export.plan([row], tmp_path / "mirror", force=True)
    assert (item.status, item.detail) == ("changed", "--force")


def test_metadata_from_an_older_schema_is_reexported(tmp_path, uploads):
    row, _ = _row(tmp_path, "m_l2_4_init0")
    assert hf_export.main(_args(tmp_path, _manifest(tmp_path, [row]), "--apply")) == 0
    [item] = hf_export.plan([row], tmp_path / "mirror")
    meta = hf_export._read_meta(item.dest)
    del meta["schema"], meta["threat_model"]
    (item.dest / "metadata.json").write_text(__import__("json").dumps(meta))

    [item] = hf_export.plan([row], tmp_path / "mirror")
    assert item.status == "changed" and "schema" in item.detail


def test_threat_model_is_reported_in_naming_units():
    # A checkpoint stores eps already rescaled by normalize_attack_epsilons.
    tm = hf_export.threat_model
    linf = tm({"advtrain": True, "attack_norm": "linf", "attack_eps": 4 / 255})
    assert (linf["eps_name"], linf["eps_internal"]) == (4.0, 4 / 255)
    l1 = tm({"advtrain": True, "attack_norm": "l1", "attack_eps": 150.0})
    assert (l1["eps_name"], l1["eps_internal"]) == (4.0, 150.0)
    assert tm({"advtrain": True, "attack_norm": "l2", "attack_eps": 2.0})["eps_name"] == 2.0
    # GradNorm trains without adversarial examples but is scored at its threat model.
    assert tm({"advtrain": False, "gradnorm": True, "attack_norm": "linf",
               "attack_eps": 1 / 255}) == {"norm": "linf", "eps_name": 1.0,
                                           "eps_internal": 1 / 255, "domain": "pixel"}
    # A clean baseline still carries attack defaults; they must not read as a threat model.
    assert tm({"advtrain": False, "gradnorm": False, "attack_norm": "linf",
               "attack_eps": 1 / 255}) is None
    assert hf_export._threat_label(tm({"advtrain": True, "attack_norm": "linf",
                                       "attack_eps": 4 / 255})) == "L∞ 4/255"


def test_missing_ema_fails_loudly_and_blocks_the_upload(tmp_path, uploads, capsys):
    good, _ = _row(tmp_path, "good_init0")
    bad, _ = _row(tmp_path, "bad_init0", ema=False)
    manifest = _manifest(tmp_path, [good, bad])

    assert hf_export.main(_args(tmp_path, manifest, "--apply", "--upload")) == 1

    mirror = tmp_path / "mirror"
    assert (mirror / "convnext_base/madry/l2/good_init0/model.safetensors").is_file()
    assert not (mirror / "convnext_base/madry/l2/bad_init0").exists() or \
        not (mirror / "convnext_base/madry/l2/bad_init0/model.safetensors").exists()
    assert "no state_dict_ema" in capsys.readouterr().err
    assert uploads == []
    assert "bad_init0" not in (mirror / "README.md").read_text()


def test_orphans_are_reported_and_left_in_place(tmp_path, uploads, capsys):
    row, _ = _row(tmp_path, "m_l2_4_init0")
    manifest = _manifest(tmp_path, [row])
    stray = tmp_path / "mirror" / "convnext_base/madry/l2/retired_init0"
    stray.mkdir(parents=True)
    (stray / "metadata.json").write_text("{}")
    (stray / "model.safetensors").write_bytes(b"x")

    assert hf_export.main(_args(tmp_path, manifest, "--apply")) == 0

    assert "ORPHAN" in capsys.readouterr().out
    assert (stray / "model.safetensors").exists()


def test_dry_run_writes_nothing(tmp_path, uploads):
    row, _ = _row(tmp_path, "m_l2_4_init0")
    manifest = _manifest(tmp_path, [row])

    assert hf_export.main(_args(tmp_path, manifest)) == 0

    assert not (tmp_path / "mirror").exists()
    assert not (tmp_path / "sha.jsonl").exists()
    assert uploads == []


def test_upload_requires_apply(tmp_path, uploads):
    row, _ = _row(tmp_path, "m_l2_4_init0")
    assert hf_export.main(_args(tmp_path, _manifest(tmp_path, [row]), "--upload")) == 2


def test_only_limits_the_export_but_not_the_index(tmp_path, uploads):
    a, _ = _row(tmp_path, "a_init0")
    b, _ = _row(tmp_path, "b_init0")
    manifest = _manifest(tmp_path, [a, b])
    assert hf_export.main(_args(tmp_path, manifest, "--apply")) == 0

    _ckpt(Path(a["target"]), seed=5)
    assert hf_export.main(_args(tmp_path, manifest, "--apply", "--only", "*a_init0*")) == 0

    readme = (tmp_path / "mirror" / "README.md").read_text()
    assert "a_init0" in readme and "b_init0" in readme
