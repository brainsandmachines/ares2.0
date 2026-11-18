import math

import pandas as pd
import pytest

from experiments.geirhos_ood.metrics import (benchmark_summary, consistency, get_short_imagename, human_metrics,
                                             model_metrics, read_trials)
from experiments.geirhos_ood.tasks import TASKS


def obs(correct, ids=None, cond="c50"):
    ids = ids or [f"n0{i}_x{i}.png" for i in range(len(correct))]
    return pd.DataFrame({"image_id": ids, "category": ["dog"] * len(correct),
                         "object_response": ["dog" if c else "cat" for c in correct],
                         "condition": [cond] * len(correct)})


def test_upstream_image_id_examples():
    assert get_short_imagename("0001_cop_s01_c30_knife_10_n03041632_7380.png") == "n03041632_7380.png"
    assert get_short_imagename("0001_ske_s01_0_boat_00_boat-0068-sketch-23.png") == "boat-0068-sketch-23.png"
    assert get_short_imagename("0001_sif_s01_0_boat_00_boat5.png") == "boat5.png"


def test_error_consistency_by_hand():
    a = obs([1, 1, 0, 0])
    b = obs([1, 0, 0, 1])
    r = consistency(a, b)
    # observed = 2/4; p1 = p2 = .5 -> expected .5 -> kappa 0
    assert r["observed_consistency"] == 0.5
    assert r["error_consistency"] == pytest.approx(0.0)
    r = consistency(obs([1, 1, 1, 0]), obs([1, 1, 1, 0]))
    assert r == {"observed_consistency": 1.0, "error_consistency": 1.0}
    # p1=.75, p2=.5, observed=.75 -> expected .5 -> kappa .5
    r = consistency(obs([1, 1, 1, 0]), obs([1, 1, 0, 0]))
    assert r["error_consistency"] == pytest.approx(0.5)


def test_consistency_requires_same_images():
    with pytest.raises(ValueError, match="different images"):
        consistency(obs([1, 0]), obs([1, 0], ids=["a.png", "b.png"]))


def test_ood_accuracy_drops_excluded_conditions_and_pairs_per_condition():
    task = TASKS["contrast"]
    model = pd.concat([obs([1, 1], ["n0a_1", "n0a_2"], "c100"), obs([1, 0], ["n0b_1", "n0b_2"], "c50")])
    model["object_response_max"] = model["object_response"]
    human = model.assign(subj="subject-01")
    out = model_metrics(model, human, task)
    assert out["n_scored_trials"] == 2           # c100 is excluded upstream
    assert out["ood_accuracy"] == 0.5
    assert out["error_consistency"] == 1.0 and out["accuracy_difference"] == 0.0
    assert out["metric_status"] == "ok" and out["n_humans"] == 1


def test_no_human_data_gives_nan_not_zero():
    out = model_metrics(obs([1, 0], cond="c50").assign(object_response_max="dog"), None, TASKS["contrast"])
    assert out["ood_accuracy"] == 0.5
    assert math.isnan(out["error_consistency"]) and out["metric_status"] == "no_human_data"


def test_human_reference_uses_unordered_pairs():
    h = pd.concat([obs([1, 1, 0, 0]).assign(subj="subject-01"), obs([1, 1, 0, 0]).assign(subj="subject-02"),
                   obs([1, 0, 0, 1]).assign(subj="subject-03")])
    out = human_metrics(h, TASKS["contrast"])
    assert out["n_humans"] == 3
    assert out["ood_accuracy"] == 0.5
    # pairs: (1,2) kappa 1, (1,3) 0, (2,3) 0
    assert out["error_consistency"] == pytest.approx(1 / 3)


def test_read_trials_parses_conditions_like_upstream(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame({"subj": ["subject-01"] * 2, "Session": [1, 1], "trial": [1, 2], "rt": [1, 1],
                  "object_response": ["car", "car"], "category": ["car", "car"], "condition": ["true", "false"],
                  "imagename": ["0001_fc_s01_true_car_35_n03100240_1.png",
                                "0002_fc_s01_false_car_35_n03100240_2.png"]}).to_csv(p, index=False)
    df = read_trials(p)
    assert list(df.condition) == ["True", "False"]
    assert "session" in df.columns and df.image_id[0] == "n03100240_1.png"


def test_benchmark_only_for_models_with_all_tasks():
    rows = pd.DataFrame([{"model_id": "a", "task_id": t, "status": "done", "ood_accuracy": 0.5,
                          "accuracy_difference": 0.0, "observed_consistency": 1.0, "error_consistency": 0.2}
                         for t in TASKS] +
                        [{"model_id": "b", "task_id": "colour", "status": "done", "ood_accuracy": 0.9,
                          "accuracy_difference": 0.0, "observed_consistency": 1.0, "error_consistency": 0.2}])
    out = benchmark_summary(rows)
    assert list(out.model_id) == ["a"] and out.ood_accuracy[0] == 0.5
