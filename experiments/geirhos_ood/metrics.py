"""model-vs-human metrics, reimplemented from upstream definitions (bethgelab/model-vs-human, master).

- 16-class decision: ImageNet softmax -> mean probability per category
  (``datasets/decision_mappings.py``, ``aggregation_function = np.mean``) -> argmax. That is
  ``data_analysis.shape_bias.mapping.aggregate_to_16(..., "mean")``. The max-aggregation
  variant is kept beside it only for the reference plot's black dot.
- OOD accuracy (``plotting/analyses.py`` SixteenClassAccuracy + ``plot.py``
  ``get_mean_over_datasets``): fraction of trials with object_response == category, pooled
  over all trials of the dataset's scored conditions (``EXCLUDE_CONDITIONS`` removed). Humans:
  pooled over every subject's trials. Benchmark: unweighted mean over datasets.
- Accuracy difference (SixteenClassAccuracyDifference, ``norm=np.square``): (acc_model - acc_human)^2.
- Observed consistency / error consistency (ErrorConsistency): trials paired by ``image_id``
  (``helper/plotting_helper.get_short_imagename``); observed = fraction where both are right or
  both wrong; expected = p1 p2 + (1-p1)(1-p2); kappa = (observed - expected) / (1 - expected),
  1.0 when observed == 1.
- Aggregation (``plot.py`` ``get_raw_benchmark_df`` + ``format_benchmark_df``): per scored
  condition (or the whole dataset when it has no conditions), model vs each human -> mean over
  conditions -> mean over humans. Humans' reference: each unordered human pair, same averaging.

Missing inputs give NaN plus a status; nothing is ever filled with 0.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.geirhos_ood.tasks import TaskSpec

TRIAL_COLUMNS = ["subj", "session", "trial", "rt", "object_response", "category", "condition", "imagename",
                 "object_response_max"]
METRIC_NAMES = ["ood_accuracy", "accuracy_difference", "observed_consistency", "error_consistency"]


def get_short_imagename(imagename: str) -> str:
    """Upstream ``plotting_helper.get_short_imagename`` (the image_id used for pairing)."""
    splits = imagename.split("_")
    if len(splits) > 1:
        name = splits[-2:]
        if name[0].startswith("n0"):
            return name[0] + "_" + name[1]
        return name[1]
    return splits[0]


def read_trials(paths: list[Path] | Path) -> pd.DataFrame:
    """Read trial CSVs the way upstream does: pandas parsing, lower-case columns, str condition."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["condition"] = df["condition"].astype(str)
    df["subj"] = df["subj"].astype(str)
    df["image_id"] = df["imagename"].apply(get_short_imagename)
    return df


def scored(df: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    conds = task.scored_conditions
    return df if conds is None else df[df["condition"].isin(conds)]


def accuracy(df: pd.DataFrame, response: str = "object_response") -> float:
    if len(df) == 0:
        return float("nan")
    return float((df[response] == df["category"]).mean())


def consistency(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, float]:
    """Observed + error consistency of two observers on the same images (upstream ErrorConsistency)."""
    d1, d2 = df1.sort_values("image_id"), df2.sort_values("image_id")
    if len(d1) != len(d2) or not d1["image_id"].reset_index(drop=True).equals(d2["image_id"].reset_index(drop=True)):
        raise ValueError(f"observers saw different images ({len(d1)} vs {len(d2)} trials)")
    c1 = (d1["object_response"] == d1["category"]).to_numpy()
    c2 = (d2["object_response"] == d2["category"]).to_numpy()
    observed = float((c1 == c2).mean())
    p1, p2 = float(c1.mean()), float(c2.mean())
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = 1.0 if observed == 1.0 else (observed - expected) / (1 - expected)
    return {"observed_consistency": observed, "error_consistency": float(kappa)}


def _units(task: TaskSpec) -> list[str | None]:
    return [None] if task.scored_conditions is None else list(task.scored_conditions)


def _pair_metrics(a: pd.DataFrame, b: pd.DataFrame, task: TaskSpec) -> dict[str, float]:
    """One observer pair: per unit (condition), then mean over units (format_benchmark_df)."""
    rows = []
    for cond in _units(task):
        x = a if cond is None else a[a["condition"] == cond]
        y = b if cond is None else b[b["condition"] == cond]
        if len(x) == 0 or len(y) == 0:
            continue
        row = {"accuracy_difference": (accuracy(x) - accuracy(y)) ** 2}
        row.update(consistency(x, y))
        rows.append(row)
    if not rows:
        return {m: float("nan") for m in METRIC_NAMES[1:]}
    return pd.DataFrame(rows).mean().to_dict()


def model_metrics(model_df: pd.DataFrame, human_df: pd.DataFrame | None, task: TaskSpec) -> dict:
    """Task-level metrics for one model; human-comparison metrics NaN without human data."""
    s = scored(model_df, task)
    out = {
        "n_trials": int(len(model_df)),
        "n_scored_trials": int(len(s)),
        "ood_accuracy": accuracy(s),
        "ood_accuracy_max": accuracy(s, "object_response_max") if "object_response_max" in s else float("nan"),
        "accuracy_difference": float("nan"),
        "observed_consistency": float("nan"),
        "error_consistency": float("nan"),
        "n_humans": 0,
        "metric_status": "no_human_data",
    }
    if human_df is None or len(human_df) == 0:
        return out
    per_human = []
    try:
        for subj, h in human_df.groupby("subj"):
            per_human.append(_pair_metrics(model_df, h, task))
    except ValueError as err:
        out["metric_status"] = f"pairing_failed: {err}"
        return out
    mean = pd.DataFrame(per_human).mean()
    out.update({k: float(mean[k]) for k in METRIC_NAMES[1:]}, n_humans=len(per_human), metric_status="ok")
    return out


def human_metrics(human_df: pd.DataFrame, task: TaskSpec) -> dict:
    """Human reference: pooled OOD accuracy + mean over unordered subject pairs."""
    out = {"ood_accuracy": accuracy(scored(human_df, task)), "n_humans": int(human_df["subj"].nunique())}
    subjects = {s: d for s, d in human_df.groupby("subj")}
    pairs = [_pair_metrics(subjects[a], subjects[b], task) for a, b in combinations(sorted(subjects), 2)]
    mean = pd.DataFrame(pairs).mean() if pairs else pd.Series(dtype=float)
    for k in METRIC_NAMES[1:]:
        out[k] = float(mean[k]) if k in mean else float("nan")
    return out


def per_condition_accuracy(df: pd.DataFrame, task: TaskSpec, response: str = "object_response") -> pd.DataFrame:
    """Accuracy at every upstream condition (excluded ones flagged), for psychometric detail."""
    if not task.data_conditions:
        return pd.DataFrame([{"condition": "all", "plot_label": "all", "excluded": False,
                              "n_trials": len(df), "accuracy": accuracy(df, response)}])
    rows = []
    for cond, label in zip(task.data_conditions, task.plotting_conditions):
        sub = df[df["condition"] == cond]
        rows.append({"condition": cond, "plot_label": label, "excluded": cond in task.excluded_conditions,
                     "n_trials": len(sub), "accuracy": accuracy(sub, response)})
    return pd.DataFrame(rows)


def benchmark_summary(task_rows: pd.DataFrame, n_tasks_required: int = 16) -> pd.DataFrame:
    """Mean over tasks per model, only for models with all required tasks measured.

    Upstream's benchmark averages 17 datasets including cue-conflict; this one covers the 16
    OOD tasks only, so it is not numerically the leaderboard figure.
    """
    ok = task_rows[task_rows["status"] == "done"]
    counts = ok.groupby("model_id")["task_id"].nunique()
    complete = counts[counts == n_tasks_required].index
    if len(complete) == 0:
        return pd.DataFrame(columns=["model_id", *METRIC_NAMES, "n_tasks"])
    agg = ok[ok["model_id"].isin(complete)].groupby("model_id")[METRIC_NAMES].mean(numeric_only=True)
    agg["n_tasks"] = n_tasks_required
    return agg.reset_index()
