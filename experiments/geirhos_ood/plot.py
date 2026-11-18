#!/usr/bin/env python3
"""One shape-bias-style page per OOD task, from ``results/task_metrics.csv``.

The page copies the last page of ``data_analysis/plots/selected_models/selected_models_aa_shape_bias.pdf``
(``all_models_bars_page`` in ``data_analysis/plot_selected_models_report.py``):
- one bar per checkpoint, grouped by family in the report's order, grey separators and the family name on top
- bar = mean-aggregation score, solid ``BAR_COLOR`` for the JSON pick, light ``OTHER_INIT_COLOR`` for the other init
- black dot = max-aggregation score
- rotated value text above each bar, "n/a" where there is no value
- bold tick labels for JSON picks, legend above the axes, y from 0 to 1.12

The score is 16-class OOD accuracy (upstream definition, excluded conditions removed), not
shape bias. The one addition is a dashed line for the pooled human accuracy, drawn when the
human raw-data exists.

    python -m experiments.geirhos_ood.plot [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from data_analysis.plot_selected_models_report import BAR_COLOR, BASELINE_COLOR, OTHER_INIT_COLOR
from experiments.geirhos_ood.models import ModelEntry, group_by_family, load_models
from experiments.geirhos_ood.tasks import TASKS, TaskSpec, task_number

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
PLOTS_DIR = EXP_DIR / "plots"
Y_LABEL = "OOD accuracy (16-class, fraction correct)"


def task_stem(task: TaskSpec) -> str:
    return f"{task_number(task.task_id):02d}_{task.task_id}"


def plotted_values(models: list[ModelEntry], task: TaskSpec, metrics: pd.DataFrame) -> pd.DataFrame:
    """The exact numbers on one task page, in x order (NaN = not measured)."""
    sub = metrics[metrics.task_id == task.task_id].set_index("model_id")
    rows = []
    for x, m in enumerate(models):
        r = sub.loc[m.model_id] if m.model_id in sub.index else None
        done = r is not None and r.get("status") == "done"
        rows.append({
            "x": x, "family": m.family, "label": m.label, "model_id": m.model_id, "in_adv_json": m.in_adv_json,
            "checkpoint": str(m.checkpoint), "status": r.get("status") if r is not None else "not_scored",
            "ood_accuracy": float(r["ood_accuracy"]) if done else np.nan,
            "ood_accuracy_max": float(r["ood_accuracy_max"]) if done else np.nan,
            "error_consistency": float(r["error_consistency"]) if done and "error_consistency" in r else np.nan,
            "n_scored_trials": int(r["n_scored_trials"]) if done else 0,
        })
    return pd.DataFrame(rows)


def draw_task_page(task: TaskSpec, models: list[ModelEntry], values: pd.DataFrame, human: dict | None,
                   watermark: str | None = None):
    """The reference's all-models bar page, for one task. Returns the figure."""
    groups = group_by_family(models)
    fig, ax = plt.subplots(figsize=(0.26 * len(models) + 3, 11))
    x = values["x"].to_numpy()
    means, maxes = values["ood_accuracy"].to_numpy(), values["ood_accuracy_max"].to_numpy()
    ax.bar(x, means, color=[BAR_COLOR if s else OTHER_INIT_COLOR for s in values["in_adv_json"]], width=0.65,
           label="OOD accuracy (mean aggregation), JSON pick")
    ax.scatter(x, maxes, color="black", zorder=3, s=22, label="OOD accuracy (max aggregation)")
    for xi, value in zip(x, means):
        ax.text(xi, (0 if np.isnan(value) else value) + 0.015, "n/a" if np.isnan(value) else f"{value:.3f}",
                ha="center", va="bottom", fontsize=6, rotation=90)
    if human and not np.isnan(human.get("ood_accuracy", np.nan)):
        ax.axhline(human["ood_accuracy"], color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
                   label=f"humans (pooled, n={human['n_humans']}): {human['ood_accuracy']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(values["label"], rotation=90, ha="center", fontsize=7)
    for tick_label, pick in zip(ax.get_xticklabels(), values["in_adv_json"]):
        if pick:
            tick_label.set_fontweight("bold")
    start = 0
    for family, ms in groups.items():
        if start:
            ax.axvline(start - 0.5, color="grey", linewidth=0.8)
        ax.text(start + len(ms) / 2 - 0.5, 1.0, family if len(ms) >= 5 else family.replace(" ", "\n"),
                ha="center", va="bottom", fontsize=8)
        start += len(ms)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(Y_LABEL, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(color=OTHER_INIT_COLOR, label="OOD accuracy (mean aggregation), other init"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, frameon=False)
    n_done = int(values["status"].eq("done").sum())
    incomplete = "" if n_done == len(values) else f"  [INCOMPLETE: {n_done}/{len(values)} models measured]"
    fig.suptitle(f"{task.label} ({task.task_id}): OOD accuracy, all models "
                 f"(bold / solid = experiment checkpoint from the JSON){incomplete}", fontsize=15)
    if watermark:
        fig.text(0.5, 0.5, watermark, fontsize=90, color="red", alpha=0.25, ha="center", va="center", rotation=20)
    fig.tight_layout()
    return fig


def draw_overview(models: list[ModelEntry], metrics: pd.DataFrame, column: str, title: str, vmin: float,
                  vmax: float, watermark: str | None = None):
    """Model x task heatmap (supplementary; not a substitute for the task pages)."""
    tasks = list(TASKS.values())
    data = np.full((len(models), len(tasks)), np.nan)
    idx = metrics[metrics.status == "done"].set_index(["model_id", "task_id"])[column]
    for r, m in enumerate(models):
        for c, t in enumerate(tasks):
            if (m.model_id, t.task_id) in idx.index:
                data[r, c] = idx.loc[(m.model_id, t.task_id)]
    fig, ax = plt.subplots(figsize=(11, 0.14 * len(models) + 3))
    im = ax.imshow(np.ma.masked_invalid(data), cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.label for t in tasks], rotation=45, ha="left", fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.label for m in models], fontsize=5)
    for tick_label, m in zip(ax.get_yticklabels(), models):
        if m.in_adv_json:
            tick_label.set_fontweight("bold")
    fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02, label=column)
    fig.suptitle(title, fontsize=13)
    if watermark:
        fig.text(0.5, 0.5, watermark, fontsize=70, color="red", alpha=0.25, ha="center", va="center", rotation=20)
    fig.tight_layout()
    return fig


def make_plots(metrics: pd.DataFrame, human: pd.DataFrame | None, models: list[ModelEntry], out_dir: Path,
               watermark: str | None = None) -> dict:
    """Write per-task figures + values, the multipage PDF and the overview. Returns a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    human_by_task = {r.task_id: r._asdict() for r in human.itertuples()} if human is not None else {}
    manifest = {"complete_tasks": [], "partial_tasks": [], "empty_tasks": [], "files": []}
    pages = []
    for task in TASKS.values():
        values = plotted_values(models, task, metrics)
        n_done = int(values["status"].eq("done").sum())
        if n_done == 0:
            manifest["empty_tasks"].append(task.task_id)
            continue
        (manifest["complete_tasks"] if n_done == len(values) else manifest["partial_tasks"]).append(task.task_id)
        stem = out_dir / task_stem(task)
        values.to_csv(stem.with_name(stem.name + "_values.csv"), index=False)
        stem.with_name(stem.name + "_values.json").write_text(json.dumps({
            "task_id": task.task_id, "label": task.label, "metric": "ood_accuracy (16-class, mean aggregation); "
            "dot = max aggregation", "scored_conditions": task.scored_conditions,
            "human": human_by_task.get(task.task_id), "models": values.to_dict(orient="records"),
        }, indent=1, default=str))
        fig = draw_task_page(task, models, values, human_by_task.get(task.task_id), watermark)
        for ext in ("pdf", "png"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=110 if ext == "png" else None)
            manifest["files"].append(str(stem.with_suffix(f".{ext}")))
        pages.append(fig)
    complete = len(manifest["complete_tasks"]) == len(TASKS)
    if pages:
        name = "geirhos_ood_16_tasks.pdf" if complete else \
            f"geirhos_ood_INCOMPLETE_{len(manifest['complete_tasks'])}of16.pdf"
        with PdfPages(out_dir / name) as pdf:
            for fig in pages:
                pdf.savefig(fig)
        manifest["files"].append(str(out_dir / name))
        with PdfPages(out_dir / "overview_heatmaps.pdf") as pdf:
            for column, title, lo, hi in (("ood_accuracy", "OOD accuracy (16-class, mean aggregation)", 0, 1),
                                          ("error_consistency", "Error consistency with humans (Cohen's kappa)",
                                           -0.2, 1)):
                fig = draw_overview(models, metrics, column, title, lo, hi, watermark)
                pdf.savefig(fig)
                plt.close(fig)
        manifest["files"].append(str(out_dir / "overview_heatmaps.pdf"))
    for fig in pages:
        plt.close(fig)
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()
    metrics = pd.read_csv(args.results_dir / "task_metrics.csv")
    human_path = args.results_dir / "human_metrics.csv"
    human = pd.read_csv(human_path) if human_path.exists() else None
    manifest = make_plots(metrics, human, load_models(), args.out_dir)
    print(f"complete: {len(manifest['complete_tasks'])}/16, partial: {manifest['partial_tasks']}, "
          f"no results: {manifest['empty_tasks']}")
    for f in manifest["files"]:
        print("  ", f)


if __name__ == "__main__":
    main()
