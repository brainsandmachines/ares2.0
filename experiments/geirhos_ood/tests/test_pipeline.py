"""Data checks, dry run, a synthetic end-to-end run on CPU, resume, and the no-download guarantee."""

import math
import socket
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import pytest
import torch

from experiments.geirhos_ood import data as D
from experiments.geirhos_ood import evaluate as E
from experiments.geirhos_ood.tasks import TASKS

from .conftest import build_root


class ConstantModel(torch.nn.Module):
    """Always answers ImageNet class 404 (airplane)."""

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 1000)
        logits[:, 404] = 10.0
        return logits


@pytest.fixture
def patched(monkeypatch, tiny_models):
    monkeypatch.setattr(E, "load_models", lambda: tiny_models)
    from types import SimpleNamespace

    import data_analysis.autoattack_array_eval as A
    monkeypatch.setattr(A, "load_model", lambda ckpt, device: (ConstantModel(), SimpleNamespace(input_size=224),
                                                               "state_dict_ema"))
    return tiny_models


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    try:
        import requests
        monkeypatch.setattr(requests, "get", boom)
        monkeypatch.setattr(requests.Session, "request", boom)
    except ImportError:
        pass


def test_check_task_statuses(tmp_path):
    t = TASKS["contrast"]
    assert D.check_task(None, t).status == D.NO_ROOT
    assert D.check_task(tmp_path, t).status == D.MISSING_DATA
    build_root(tmp_path, ["contrast"], with_humans=False)
    assert D.check_task(tmp_path, t).status == D.READY_NO_HUMAN
    build_root(tmp_path, ["contrast"])
    s = D.check_task(tmp_path, t)
    assert s.status == D.READY and s.n_images == 16 and "1280" in s.detail
    bad = tmp_path / "datasets" / "contrast" / "dnn" / "session-1" / "badname.png"
    bad.write_bytes(b"")
    assert D.check_task(tmp_path, t).status == D.BAD_LAYOUT


def test_edge_layout_uses_category_folders(tmp_path):
    build_root(tmp_path, ["edge"])
    s = D.check_task(tmp_path, TASKS["edge"])
    assert s.status == D.READY
    sample = D.OODTaskDataset(tmp_path, TASKS["edge"]).samples[0]
    assert sample["condition"] == "NaN" and sample["category"] in ("airplane", "bear", "knife", "oven")


def test_env_root_and_no_modelvshumandir(monkeypatch, tmp_path):
    monkeypatch.delenv(D.ENV_DATA_ROOT, raising=False)
    monkeypatch.setenv("MODELVSHUMANDIR", str(tmp_path))
    assert D.resolve_data_root(None) is None
    monkeypatch.setenv(D.ENV_DATA_ROOT, str(tmp_path))
    assert D.resolve_data_root(None) == tmp_path


def test_dry_run_reports_every_cell(patched, tmp_path, capsys, no_network):
    root = build_root(tmp_path / "root", ["colour"])
    out = tmp_path / "out"
    E.main(["--dry-run", "--data-root", str(root), "--out-dir", str(out)])
    table = pd.read_csv(out / "dry_run_status.csv")
    assert len(table) == len(patched) * 16
    assert set(table[table.task_id == "colour"].status) == {"ready"}
    assert set(table[table.task_id != "colour"].status) == {"missing_data"}
    assert "colour" in capsys.readouterr().out
    assert not (out / "trials").exists()          # dry run loads nothing, writes no trials


def test_missing_checkpoint_is_reported(patched, tmp_path):
    patched[0].checkpoint.unlink()
    table, _, _ = E.status_table(patched, [TASKS["colour"]], build_root(tmp_path, ["colour"]), tmp_path / "o")
    assert table.set_index("model_id").loc[patched[0].model_id, "status"] == "missing_checkpoint"


def test_missing_data_never_downloads_and_never_scores_zero(patched, tmp_path, monkeypatch, no_network):
    monkeypatch.delitem(sys.modules, "modelvshuman", raising=False)
    out = tmp_path / "out"
    E.main(["--data-root", str(tmp_path / "empty"), "--out-dir", str(out), "--device", "cpu"])
    rows = pd.read_csv(out / "task_metrics.csv")
    assert set(rows.status) == {"missing_data"}
    assert "ood_accuracy" not in rows or rows.ood_accuracy.isna().all()
    assert "modelvshuman" not in sys.modules
    assert not (tmp_path / "empty").exists()      # nothing was created at the data root


def test_sources_have_no_downloader():
    src = Path(E.__file__).parent
    for f in src.glob("*.py"):
        text = f.read_text()
        for needle in ("import requests", "urlopen", "urlretrieve", "import modelvshuman", "from modelvshuman",
                       "torch.hub", "wget", "curl"):
            assert needle not in text, f"{f.name} contains {needle!r}"


def test_end_to_end_cpu_resume_and_metrics(patched, tmp_path, no_network):
    root = build_root(tmp_path / "root", ["contrast", "sketch", "edge"])
    out = tmp_path / "out"
    args = ["--data-root", str(root), "--out-dir", str(out), "--device", "cpu", "--num-workers", "0",
            "--batch-size", "4", "--tasks", "contrast", "sketch", "edge"]
    E.main(args)
    trial = E.trial_path(out, patched[0], TASKS["contrast"])
    df = pd.read_csv(trial)
    assert list(df.columns) == E.TRIAL_COLUMNS and len(df) == 16
    assert set(df.object_response) == {"airplane"}
    metrics = pd.read_csv(out / "task_metrics.csv")
    done = metrics[metrics.status == "done"]
    assert len(done) == len(patched) * 3
    contrast = done[done.task_id == "contrast"].iloc[0]
    expected = (df[df.condition.isin(TASKS["contrast"].scored_conditions)].category == "airplane").mean()
    assert contrast.ood_accuracy == pytest.approx(expected)
    assert contrast.metric_status == "ok" and not math.isnan(contrast.error_consistency)
    assert (out / "human_metrics.csv").exists() and list((out / "configs").glob("run_*.json"))

    mtime = trial.stat().st_mtime_ns
    E.main(args)                                   # resume: complete files are skipped
    assert trial.stat().st_mtime_ns == mtime
    E.main(args + ["--force"])
    assert trial.stat().st_mtime_ns != mtime
