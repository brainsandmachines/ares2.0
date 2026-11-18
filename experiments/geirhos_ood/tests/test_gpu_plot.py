"""GPU guard (mocked CUDA) and plotting from synthetic fixtures (watermarked, never real results)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.geirhos_ood.gpu import GB, GPUBusyError, budget_fraction, check_free_memory, forward_with_backoff
from experiments.geirhos_ood.plot import make_plots, plotted_values
from experiments.geirhos_ood.tasks import TASKS


def test_refuses_when_memory_is_short():
    with pytest.raises(GPUBusyError):
        check_free_memory(4, 3, mem_get_info=lambda: (6 * GB, 24 * GB))
    assert check_free_memory(4, 3, mem_get_info=lambda: (17 * GB, 24 * GB)) == (17.0, 24.0)


def test_budget_fraction():
    assert budget_fraction(6, 24 * GB) == 0.25
    assert budget_fraction(48, 24 * GB) == 1.0
    with pytest.raises(ValueError):
        budget_fraction(0, 24 * GB)


def test_oom_backoff_gives_identical_output():
    w = torch.randn(3, 5)
    calls = []

    def fn(x):
        calls.append(x.shape[0])
        if x.shape[0] > 4:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return x @ w.T

    x = torch.randn(16, 5)
    out = forward_with_backoff(fn, x, min_batch=2)
    assert torch.allclose(out, x @ w.T)
    assert max(c for c in calls[1:]) <= 8 and 4 in calls


def test_oom_at_min_batch_reraises():
    def fn(x):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        forward_with_backoff(fn, torch.zeros(8, 1), min_batch=4)


def synthetic_metrics(models, tasks, rng, drop=()):
    rows = []
    for m in models:
        for t in tasks:
            if (m.model_id, t) in drop:
                rows.append({"model_id": m.model_id, "task_id": t, "status": "not_run"})
                continue
            a = float(rng.uniform(0.2, 0.9))
            rows.append({"model_id": m.model_id, "task_id": t, "status": "done", "ood_accuracy": a,
                         "ood_accuracy_max": a + 0.02, "error_consistency": float(rng.uniform(0, 0.5)),
                         "n_scored_trials": 100})
    return pd.DataFrame(rows)


def test_plots_complete_set(tiny_models, tmp_path):
    rng = np.random.default_rng(0)
    metrics = synthetic_metrics(tiny_models, list(TASKS), rng)
    human = pd.DataFrame([{"task_id": t, "ood_accuracy": 0.8, "n_humans": 4} for t in TASKS])
    manifest = make_plots(metrics, human, tiny_models, tmp_path, watermark="SYNTHETIC")
    assert len(manifest["complete_tasks"]) == 16
    assert (tmp_path / "geirhos_ood_16_tasks.pdf").exists()
    assert (tmp_path / "03_high-pass.png").exists() and (tmp_path / "03_high-pass.pdf").exists()
    values = pd.read_csv(tmp_path / "01_colour_values.csv")
    expected = metrics[metrics.task_id == "colour"].set_index("model_id").ood_accuracy
    assert np.allclose(values.set_index("model_id").ood_accuracy.loc[expected.index], expected)
    assert json.loads((tmp_path / "01_colour_values.json").read_text())["human"]["ood_accuracy"] == 0.8


def test_plots_incomplete_are_labelled_and_not_filled(tiny_models, tmp_path):
    rng = np.random.default_rng(1)
    metrics = synthetic_metrics(tiny_models, ["colour", "edge"], rng, drop={(tiny_models[0].model_id, "edge")})
    manifest = make_plots(metrics, None, tiny_models, tmp_path, watermark="SYNTHETIC")
    assert manifest["complete_tasks"] == ["colour"] and manifest["partial_tasks"] == ["edge"]
    assert len(manifest["empty_tasks"]) == 14
    assert (tmp_path / "geirhos_ood_INCOMPLETE_1of16.pdf").exists()
    assert not (tmp_path / "geirhos_ood_16_tasks.pdf").exists()
    vals = plotted_values(tiny_models, TASKS["edge"], metrics)
    assert np.isnan(vals.ood_accuracy[0]) and vals.status[0] == "not_run"
