#!/usr/bin/env python3
"""PDF report for the selected models: AutoAttack sweep + shape bias, one page per family.

Page 1 is the clean baselines' shape bias (they have no AA sweep). Family pages show both
inits where a second one exists; the zoo JSON's pick is in bold (solid bar). The last two
pages are shape bias for all models: a family x eps heatmap (JSON picks) and one bar per
checkpoint.

Each family page puts the AutoAttack comparation heatmap (drawn by the existing
``plot_autoattack_comparation.py``, from each model's own experiment checkpoint CSV) next
to the shape-bias scores of the same checkpoints. A last page shows shape bias for every
model. Shape-bias scores come from ``shape_bias_results_botero.csv``, written by
``shape_bias_selected.py``.

python -m data_analysis.plot_selected_models_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from aa_sweep.config import EPS_INPUTS
from data_analysis.plot_autoattack_comparation import (
    ATTACK_NORMS,
    accuracy_for_eval_eps,
    csv_name_for_checkpoint_kind,
    load_autoattack_results,
    plot_comparation,
)
from data_analysis.selected_models import FAMILIES, Selected, resolve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAPE_BIAS_CSV = PROJECT_ROOT / "data_analysis" / "shape_bias_results_botero.csv"
OUT_DIR = PROJECT_ROOT / "data_analysis" / "plots" / "selected_models"
PDF_NAME = "selected_models_aa_shape_bias.pdf"

# The AA sweep's goal grid (eps 12 was dropped from it), plus 0 for clean accuracy.
EVAL_EPS_ORDER = [0.0, *EPS_INPUTS]
PAGE_SIZE = (20.0, 10.0)
SUMMARY_EPS = [1.0, 2.0, 4.0, 6.0, 8.0]
SUMMARY_EPS_LABELS = ["1", "2", "4", "cont→6", "cont→8"]
BAR_COLOR = "#4C72B0"
OTHER_INIT_COLOR = "#A9BEDC"
BASELINE_COLOR = "#C44E52"


def label_for(sel: Selected) -> str:
    return f"{sel.key[len(sel.arch) + 1:].replace('_', '-')} [{sel.kind}]"


def load_shape_bias(csv_path: Path) -> pd.DataFrame:
    """Latest row per (run dir, checkpoint kind), indexed that way."""
    df = pd.read_csv(csv_path).sort_values("time")
    return df.drop_duplicates(["model_name", "checkpoint"], keep="last").set_index(["model_name", "checkpoint"])


def shape_bias_of(sb: pd.DataFrame, sel: Selected, column: str = "score") -> float:
    key = (sel.run_dir, sel.kind)
    return float(sb.loc[key, column]) if key in sb.index else float("nan")


def partial_aa_column(models_root: Path, entry: tuple[str, str, str]) -> dict[str, np.ndarray]:
    """An incompletely swept model: the cells it has, NaN (drawn "n/a") for the rest."""
    csv_path = models_root / entry[0] / entry[2]
    df = pd.read_csv(csv_path)
    df["attack_norm"] = df["attack_norm"].astype(str).str.lower()
    column = {}
    for attack_norm in ATTACK_NORMS:
        values = []
        for eval_eps in EVAL_EPS_ORDER:
            try:
                values.append(accuracy_for_eval_eps(df[df["attack_norm"] == attack_norm], csv_path,
                                                    attack_norm, eval_eps))
            except ValueError:
                values.append(np.nan)
        column[attack_norm] = np.array(values)[:, None]
    return column


def plot_aa_png(family: str, models: list[Selected], out_png: Path) -> None:
    """One AA comparation heatmap via the existing plotting functions, per-model checkpoint kind."""
    models_root = models[0].checkpoint.parent.parent
    entries = [(s.run_dir, label_for(s), csv_name_for_checkpoint_kind(s.kind)) for s in models]
    columns = []
    for entry in entries:
        try:
            columns.append(load_autoattack_results(models_root, [entry], EVAL_EPS_ORDER))
        except ValueError as err:
            print(f"WARNING: incomplete AA sweep, missing cells shown as n/a: {err}")
            columns.append(partial_aa_column(models_root, entry))
    data_by_norm = {norm: np.hstack([c[norm] for c in columns]) for norm in ATTACK_NORMS}
    plot_comparation(data_by_norm, [label for _m, label, _c in entries], EVAL_EPS_ORDER, family, out_png,
                     bold_labels=[label_for(s) for s in models if s.in_json])


def draw_shape_bias_bars(ax, models: list[Selected], labels: list[str], baseline: Selected | None,
                         sb: pd.DataFrame) -> None:
    x = np.arange(len(models))
    means = [shape_bias_of(sb, s) for s in models]
    maxes = [shape_bias_of(sb, s, "score_max") for s in models]
    two_inits = not all(s.in_json for s in models)
    ax.bar(x, means, color=[BAR_COLOR if s.in_json else OTHER_INIT_COLOR for s in models], width=0.65,
           label="shape bias (mean aggregation)" + (", JSON pick" if two_inits else ""))
    ax.scatter(x, maxes, color="black", zorder=3, s=22, label="shape bias (max aggregation)")
    for xi, value in zip(x, means):
        ax.text(xi, value + 0.015, "n/a" if np.isnan(value) else f"{value:.3f}", ha="center", fontsize=10)
    base = shape_bias_of(sb, baseline) if baseline is not None else float("nan")
    if not np.isnan(base):
        ax.axhline(base, color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
                   label=f"clean baseline {label_for(baseline)}: {base:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    for tick_label, sel in zip(ax.get_xticklabels(), models):
        if sel.in_json and two_inits:
            tick_label.set_fontweight("bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("shape bias (fraction of shape decisions)", fontsize=11)
    ax.set_title("Shape bias (Geirhos cue-conflict, 1280 stimuli)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    handles, _labels = ax.get_legend_handles_labels()
    if two_inits:
        handles.append(Patch(color=OTHER_INIT_COLOR, label="shape bias (mean aggregation), other init"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)


def family_page(pdf: PdfPages, family: str, models: list[Selected], baseline: Selected,
                sb: pd.DataFrame, aa_png: Path) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    image = plt.imread(aa_png)
    grid = fig.add_gridspec(1, 2, width_ratios=[image.shape[1] / image.shape[0], 1.15], wspace=0.05)
    ax_aa = fig.add_subplot(grid[0])
    ax_aa.imshow(image)
    ax_aa.axis("off")
    draw_shape_bias_bars(fig.add_subplot(grid[1]), models, [label_for(s) for s in models], baseline, sb)
    subtitle = ("bold = experiment checkpoint from models_for_adversarial_experiments.json; "
                "light = other init from models_for_blackbox_experiments.json"
                if not all(s.in_json for s in models)
                else "experiment checkpoints from models_for_adversarial_experiments.json (only init1 exists)")
    fig.suptitle(f"{family}\n{subtitle}", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.27)
    pdf.savefig(fig)
    plt.close(fig)


def baselines_page(pdf: PdfPages, baselines: dict[str, Selected], sb: pd.DataFrame) -> None:
    """Clean (non-adversarially-trained) baselines: shape bias only, they have no AA sweep."""
    models = list(baselines.values())
    fig, ax = plt.subplots(figsize=(11, 8.5))
    draw_shape_bias_bars(ax, models, [f"{s.key} [{s.kind}]" for s in models], None, sb)
    fig.suptitle("Clean baselines (no adversarial training, no AA sweep)", fontsize=15)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def all_models_bars_page(pdf: PdfPages, families: dict[str, list[Selected]], baselines: dict[str, Selected],
                         sb: pd.DataFrame) -> None:
    """Every checkpoint in the report on one axis: x = model name, y = shape bias, families separated."""
    groups = {"clean baselines": list(baselines.values()), **families}
    models = [m for ms in groups.values() for m in ms]
    fig, ax = plt.subplots(figsize=(0.26 * len(models) + 3, 11))
    draw_shape_bias_bars(ax, models, [f"{m.key} [{m.kind}]" for m in models], None, sb)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(7)
        tick_label.set_rotation(90)
        tick_label.set_ha("center")
    for text in ax.texts:
        text.set_fontsize(6)
        text.set_rotation(90)
        text.set_va("bottom")
    start = 0
    for family, ms in groups.items():
        if start:
            ax.axvline(start - 0.5, color="grey", linewidth=0.8)
        ax.text(start + len(ms) / 2 - 0.5, 1.0, family if len(ms) >= 5 else family.replace(" ", "\n"),
                ha="center", va="bottom", fontsize=8)
        start += len(ms)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_title("")
    legend = ax.get_legend()
    ax.legend(legend.legend_handles, [t.get_text() for t in legend.get_texts()], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Shape bias, all models (bold / solid = experiment checkpoint from the JSON)", fontsize=15)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf: PdfPages, families: dict[str, list[Selected]], baselines: dict[str, Selected],
                 sb: pd.DataFrame) -> None:
    columns = ["clean\nbaseline", *SUMMARY_EPS_LABELS]
    data = np.full((len(families), len(columns)), np.nan)
    for row, (family, models) in enumerate(families.items()):
        data[row, 0] = shape_bias_of(sb, baselines[FAMILIES[family][1]])
        for sel in models:
            if sel.in_json:
                data[row, 1 + SUMMARY_EPS.index(float(sel.eps))] = shape_bias_of(sb, sel)

    fig, ax = plt.subplots(figsize=(11, 13))
    im = ax.imshow(np.ma.masked_invalid(data), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if not np.isnan(data[r, c]):
                ax.text(c, r, f"{data[r, c]:.3f}", ha="center", va="center", fontsize=10,
                        color="white" if data[r, c] < 0.6 else "black")
    ax.axvline(0.5, color="white", linewidth=4)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=11)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(list(families), fontsize=11)
    ax.set_xlabel("training eps (l1 eps shown in eps_input units)", fontsize=11)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", top=False, left=False)
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label="shape bias (mean aggregation)")
    fig.suptitle("Shape bias — all selected models (experiment checkpoints)", fontsize=15)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape-bias-csv", type=Path, default=SHAPE_BIAS_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    families, baselines = resolve()
    sb = load_shape_bias(args.shape_bias_csv)
    missing = [f"{s.run_dir}/{s.kind}" for s in [*baselines.values(), *(s for ss in families.values() for s in ss)]
               if (s.run_dir, s.kind) not in sb.index]
    if missing:
        print(f"WARNING: no shape bias for {len(missing)} checkpoints: {', '.join(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = args.out_dir / PDF_NAME
    with PdfPages(out_pdf) as pdf:
        baselines_page(pdf, baselines, sb)
        for family, models in families.items():
            aa_png = args.out_dir / f"aa_{family.replace(' ', '_')}.png"
            plot_aa_png(family, models, aa_png)
            family_page(pdf, family, models, baselines[FAMILIES[family][1]], sb, aa_png)
        summary_page(pdf, families, baselines, sb)
        all_models_bars_page(pdf, families, baselines, sb)
    print(f"Saved {out_pdf} ({len(families) + 3} pages)")


if __name__ == "__main__":
    main()
