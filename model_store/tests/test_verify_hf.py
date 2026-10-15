"""verify_hf: the Hub check catches a Hub/mirror/zoo mismatch, the forward check catches
published weights that are not the checkpoint's EMA, and the zoo facts read the right AA row.

The Hub is a fake ``list_repo_tree`` built from the mirror files (LFS sha256 for the weights,
git blob id for metadata.json); the model builder is a tiny module whose state_dict keys match
the test checkpoints of test_hf_export, so the real torch/safetensors path runs.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from model_store import hf_export, verify_hf
from model_store.tests.test_hf_export import _args, _manifest, _row


class TinyModel(torch.nn.Module):
    """state_dict keys stem.0.weight (4x3) and head.bias (4), like the test checkpoints."""

    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(torch.nn.Linear(3, 4, bias=False))
        self.head = torch.nn.Module()
        self.head.bias = torch.nn.Parameter(torch.zeros(4))

    def forward(self, x):
        return self.stem(x.mean(dim=(2, 3))) + self.head.bias


class FakeApi:
    def __init__(self, files):
        self.files = files

    def list_repo_tree(self, repo, recursive=True):
        return self.files


def _hub_from_mirror(mirror: Path) -> list:
    files = []
    for p in sorted(mirror.rglob("*")):
        if not p.is_file() or ".cache" in p.parts:
            continue
        rel = str(p.relative_to(mirror))
        data = p.read_bytes()
        lfs = (SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
               if p.name == hf_export.WEIGHTS_FILE else None)
        files.append(SimpleNamespace(path=rel, blob_id=verify_hf.git_blob_sha1(p), lfs=lfs))
    return files


def _aa_csv(target: Path, kind: str, epoch: int, rows: list[tuple[str, float, float]]) -> None:
    path = target.parent / verify_hf.CSV_FOR_KIND[kind]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attack_norm", "epsilon_input", "epoch", "clean_acc", "robust_acc"])
        for norm, eps, robust in rows:
            w.writerow([norm, eps, epoch, 71.5, robust])


@pytest.fixture
def exported(tmp_path, monkeypatch):
    """One zoo model, exported to a tmp mirror, with an AA CSV at its trained threat."""
    monkeypatch.setattr(hf_export, "upload", lambda *a: 0)
    monkeypatch.setattr(verify_hf, "build_model", lambda meta, args: TinyModel())
    row, _ = _row(tmp_path, "m_l2_4_init0")
    _aa_csv(Path(row["target"]), "last", 7, [("l2", 4.0, 33.25), ("linf", 4.0, 10.0)])
    manifest = _manifest(tmp_path, [row])
    assert hf_export.main(_args(tmp_path, manifest, "--apply")) == 0
    return SimpleNamespace(row=row, manifest=manifest, mirror=tmp_path / "mirror",
                           dest=tmp_path / "mirror" / "convnext_base/madry/l2/m_l2_4_init0")


def _verify(tmp_path, exported, monkeypatch, files, *extra) -> int:
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi(files))
    return verify_hf.main(["--manifest", str(exported.manifest), "--mirror", str(exported.mirror),
                           "--hash-cache", str(tmp_path / "sha.jsonl"),
                           "--excludes", str(tmp_path / "none.csv"),
                           "--out-csv", str(tmp_path / "report.csv"),
                           "--out-md", str(tmp_path / "list.md"), "--device", "cpu", *extra])


def _report(tmp_path) -> list[dict]:
    with (tmp_path / "report.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_trained_threat_uses_zoo_norm_and_last_eps():
    assert verify_hf.trained_threat({"model": "convnext_base_l2_4_init1", "norm": "l2"}) == ("l2", 4.0)
    assert verify_hf.trained_threat(
        {"model": "convnext_base_linftrades_cont4to8_init0", "norm": "linf"}) == ("linf", 8.0)
    # GradNorm is named after its penalty norm but scored at the zoo's norm.
    assert verify_hf.trained_threat(
        {"model": "convnext_base_gradnorm_l1_2_init0", "norm": "linf"}) == ("linf", 2.0)
    assert verify_hf.trained_threat({"model": "convnext_base_baseline_init0", "norm": ""}) is None


def test_matching_hub_passes_and_lists_the_model(tmp_path, exported, monkeypatch):
    rc = _verify(tmp_path, exported, monkeypatch, _hub_from_mirror(exported.mirror), "--forward")

    assert rc == 0
    [r] = _report(tmp_path)
    assert (r["hf_status"], r["threat"], r["aa"], r["clean_aa1k"]) == ("on-hub", "l2@4", "33.25", "71.5")
    assert r["aa_epoch_ok"] == "True" and r["forward"] == "True" and r["problems"] == ""
    md = (tmp_path / "list.md").read_text()
    assert "| m_l2_4_init0 | l2@4 | last | 7 |" in md and "forward ✅" in md


def test_hub_weights_that_differ_from_the_mirror_fail(tmp_path, exported, monkeypatch):
    files = _hub_from_mirror(exported.mirror)
    for f in files:
        if f.lfs is not None:
            f.lfs = SimpleNamespace(sha256="0" * 64)

    assert _verify(tmp_path, exported, monkeypatch, files) == 1
    assert "Hub weights sha256 != metadata" in _report(tmp_path)[0]["problems"]


def test_published_weights_not_from_the_checkpoint_fail_the_forward_check(
        tmp_path, exported, monkeypatch):
    weights = exported.dest / hf_export.WEIGHTS_FILE
    tensors = load_file(str(weights))
    tensors["head.bias"] = tensors["head.bias"] + 1.0
    save_file(tensors, str(weights))  # the Hub listing below is rebuilt, so sha checks see it

    assert _verify(tmp_path, exported, monkeypatch, _hub_from_mirror(exported.mirror),
                   "--forward") == 1
    [r] = _report(tmp_path)
    assert r["forward"] == "False" and "logits differ" in r["forward_detail"]


def test_excluded_model_on_the_hub_and_unexpected_dirs_are_reported(tmp_path, exported, monkeypatch):
    excludes = tmp_path / "ex.csv"
    excludes.write_text("pattern,reason\nm_l2_4_init0,collapsed\n")
    files = _hub_from_mirror(exported.mirror)
    files.append(SimpleNamespace(path="swin_b/madry/l2/ghost/model.safetensors", blob_id="x",
                                 lfs=SimpleNamespace(sha256="y")))
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi(files))

    rc = verify_hf.main(["--manifest", str(exported.manifest), "--mirror", str(exported.mirror),
                         "--hash-cache", str(tmp_path / "sha.jsonl"), "--excludes", str(excludes),
                         "--out-csv", str(tmp_path / "report.csv")])

    assert rc == 1
    assert _report(tmp_path)[0]["hf_status"] == "excluded-but-on-hub"


def test_aa_rows_from_another_epoch_are_a_problem(tmp_path, exported, monkeypatch):
    _aa_csv(Path(exported.row["target"]), "last", 3, [("l2", 4.0, 50.0)])

    assert _verify(tmp_path, exported, monkeypatch, _hub_from_mirror(exported.mirror)) == 1
    [r] = _report(tmp_path)
    assert r["aa_epoch_ok"] == "False" and "not all from the linked checkpoint" in r["problems"]


def test_model_not_on_the_hub_is_listed_as_not_uploaded(tmp_path, exported, monkeypatch):
    assert _verify(tmp_path, exported, monkeypatch, []) == 0
    assert _report(tmp_path)[0]["hf_status"] == "not-uploaded"
