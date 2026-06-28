#!/usr/bin/env python3
"""Plot AutoAttack robustness comparison heatmaps for selected models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ares.utils.epsilon_schedule import DEFAULT_PLOT_EPS_INPUTS, L1_EPS_MULTIPLIER

# to run this file, use this exmaple:
# python data_analysis/plot_autoattack_comparation.py     --plot_name "v1_models"      --models     convnext_small_v1_clean_init1     convnext_small_v1_noise_init1    convnext_small_v1_clean_l2_4_init1      convnext_small_v1_clean_l2_40_init1

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit these constants to change the figure without touching the data logic.
MODELS_ROOT = PROJECT_ROOT / "results" / "models"
OUTPUT_DIR = PROJECT_ROOT / "data_analysis" / "plots" / "AutoAttack"
CSV_NAME = "autoattack_sweep_results.csv"
CHECKPOINT_KINDS = ("best", "last", "advbest")
CHECKPOINT_KIND_SUFFIX = {"best": "", "last": "last", "advbest": "advbest"}
ATTACK_NORMS = ["l1", "l2", "linf"]
EVAL_EPS_ORDER = list(DEFAULT_PLOT_EPS_INPUTS)

TITLE_TEMPLATE = "{plot_name}"
COLORBAR_LABEL = "Accuracy (%)"
CMAP = "plasma"
VMIN = 0.0
VMAX = 100.0
DPI = 200
ANNOTATION_FMT = "{:.1f}%"
ANNOTATION_FONTSIZE = 10
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 13
TEXT_DARK_THRESHOLD = 20.0
GRID_COLOR = "white"
GRID_LINEWIDTH = 1.0
MIN_FIG_WIDTH = 4.3
FIG_WIDTH_PER_MODEL = 0.75
FIG_WIDTH_PADDING = 1.8
FIG_HEIGHT = 8.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AutoAttack robustness comparation heatmaps.")
    parser.add_argument("--plot_name", required=True, help="Name used in the figure title and output filenames.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model directory names to compare, in x-axis order.",
    )
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT, help="Root directory containing model folders.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory for plot files.")
    parser.add_argument(
        "--checkpoint-kind",
        choices=["best", "last", "advbest", "all"],
        default="best",
        help="Checkpoint CSV to plot: best, last, advbest, or all available kinds.",
    )
    parser.add_argument(
        "--eval-eps-order",
        default=",".join(str(eps) for eps in EVAL_EPS_ORDER),
        help="Comma-separated epsilon_input values to show on the y-axis, including 0 for clean accuracy.",
    )
    parser.add_argument("--svg", action="store_true", help="Also save an SVG version of the plot.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving.")
    args = parser.parse_args()

    if not args.plot_name.strip():
        parser.error("--plot_name must be a non-empty string.")
    if not args.models:
        parser.error("--models must include at least one model name.")
    args.models = [model.strip() for model in args.models if model.strip()]
    if not args.models:
        parser.error("--models must include at least one non-empty model name.")
    args.eval_eps_order = [float(part.strip()) for part in args.eval_eps_order.split(",") if part.strip()]
    if not args.eval_eps_order:
        parser.error("--eval-eps-order must include at least one epsilon.")

    return args


def model_name_to_label(model_name: str) -> str:
    prefix = "convnext_small_"
    label = model_name[len(prefix) :] if model_name.startswith(prefix) else model_name
    return label.replace("_", "-")


def csv_name_for_checkpoint_kind(checkpoint_kind: str) -> str:
    suffix = CHECKPOINT_KIND_SUFFIX[checkpoint_kind]
    if not suffix:
        return CSV_NAME
    path = Path(CSV_NAME)
    return f"{path.stem}_{suffix}{path.suffix}"


def model_entries(models_root: Path, models: list[str], checkpoint_kind: str) -> list[tuple[str, str, str]]:
    entries = []
    if checkpoint_kind == "all":
        for model_name in models:
            base_label = model_name_to_label(model_name)
            for kind in CHECKPOINT_KINDS:
                csv_name = csv_name_for_checkpoint_kind(kind)
                if (models_root / model_name / csv_name).exists():
                    entries.append((model_name, f"{base_label}-{kind}", csv_name))
        if not entries:
            raise FileNotFoundError("No AutoAttack CSVs found for --checkpoint-kind all.")
        return entries

    csv_name = csv_name_for_checkpoint_kind(checkpoint_kind)
    missing = [model for model in models if not (models_root / model / csv_name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {checkpoint_kind} AutoAttack CSV {csv_name} for: {', '.join(missing)}")
    return [(model_name, model_name_to_label(model_name), csv_name) for model_name in models]


def require_columns(df: pd.DataFrame, csv_path: Path, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")


def accuracy_for_eval_eps(df_norm: pd.DataFrame, csv_path: Path, attack_norm: str, eval_eps: float) -> float:

    if eval_eps == 0:
        clean_values = pd.to_numeric(df_norm["clean_acc"], errors="coerce").dropna().unique()
        if len(clean_values) == 0:
            raise ValueError(f"{csv_path} has no clean_acc values for {attack_norm} rows.")
        return float(np.mean(clean_values))

    epsilon_input = pd.to_numeric(df_norm["epsilon_input"], errors="coerce")
    rows = df_norm[np.isclose(epsilon_input, float(eval_eps))]
    if rows.empty:
        raise ValueError(f"{csv_path} has no {attack_norm} row with epsilon_input={eval_eps}.")
    robust_values = pd.to_numeric(rows["robust_acc"], errors="coerce").dropna()
    if robust_values.empty:
        raise ValueError(f"{csv_path} has no robust_acc value for {attack_norm} epsilon_input={eval_eps}.")
    return float(robust_values.mean())


def load_autoattack_results(
    models_root: Path,
    entries: list[tuple[str, str, str]],
    eval_eps_order: list[float],
) -> dict[str, np.ndarray]:
    
    data_by_norm = {
        attack_norm: np.full((len(eval_eps_order), len(entries)), np.nan, dtype=float) for attack_norm in ATTACK_NORMS
    }

    for col_idx, (model_name, _label, csv_name) in enumerate(entries):
        csv_path = models_root / model_name / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing AutoAttack sweep CSV for model {model_name!r}: {csv_path}")

        df = pd.read_csv(csv_path)
        require_columns(df, csv_path, ["attack_norm", "epsilon_input", "clean_acc", "robust_acc"])
        df["attack_norm"] = df["attack_norm"].astype(str).str.lower()

        for attack_norm in ATTACK_NORMS:
            df_norm = df[df["attack_norm"] == attack_norm].copy()
            if df_norm.empty:
                raise ValueError(f"{csv_path} has no rows with attack_norm={attack_norm!r} for model {model_name!r}.")

            for row_idx, eval_eps in enumerate(eval_eps_order):
                data_by_norm[attack_norm][row_idx, col_idx] = accuracy_for_eval_eps(
                    df_norm,
                    csv_path,
                    attack_norm,
                    eval_eps,
                )

    return data_by_norm


def trim_float(value: float) -> str:
    return f"{float(value):g}"


def ytick_labels_for_norm(attack_norm: str, eval_eps_order: list[float]) -> list[str]:
    if attack_norm == "l1":
        return [trim_float(eps * L1_EPS_MULTIPLIER) for eps in eval_eps_order]
    if attack_norm == "linf":
        return ["0" if eps == 0 else f"{trim_float(eps)}/255" for eps in eval_eps_order]
    return [trim_float(eps) for eps in eval_eps_order]


def plot_comparation(
    data_by_norm: dict[str, np.ndarray],
    model_labels: list[str],
    eval_eps_order: list[float],
    plot_name: str,
    out_png: Path,
    out_svg: Path | None = None,
    show: bool = False,
) -> None:

    fig_width = max(MIN_FIG_WIDTH, FIG_WIDTH_PER_MODEL * len(model_labels) + FIG_WIDTH_PADDING)
    fig, axes = plt.subplots(
        len(ATTACK_NORMS),
        1,
        figsize=(fig_width, FIG_HEIGHT),
        constrained_layout=True,
        sharex=True,
    )

    images = []
    for ax, attack_norm in zip(axes, ATTACK_NORMS):
        data = data_by_norm[attack_norm]
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=CMAP, vmin=VMIN, vmax=VMAX)
        images.append(im)

        ax.set_yticks(range(len(eval_eps_order)))
        ax.set_yticklabels(ytick_labels_for_norm(attack_norm, eval_eps_order), fontsize=TICK_FONTSIZE)
        ax.set_ylabel(f"{attack_norm} eval eps", fontsize=LABEL_FONTSIZE)

        ax.set_xticks(np.arange(-0.5, len(model_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(eval_eps_order), 1), minor=True)
        ax.grid(which="minor", color=GRID_COLOR, linestyle="-", linewidth=GRID_LINEWIDTH)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):
                value = float(data[row_idx, col_idx])
                text_color = "white" if value < TEXT_DARK_THRESHOLD else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    ANNOTATION_FMT.format(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=ANNOTATION_FONTSIZE,
                )

    axes[-1].set_xticks(range(len(model_labels)))
    axes[-1].set_xticklabels(model_labels, fontsize=TICK_FONTSIZE, rotation=35, ha="right")
    axes[-1].set_xlabel("Model", fontsize=LABEL_FONTSIZE)
    fig.suptitle(TITLE_TEMPLATE.format(plot_name=plot_name), fontsize=TITLE_FONTSIZE, x=0.48)

    cbar = fig.colorbar(images[-1], ax=axes, shrink=0.9, pad=0.03)
    cbar.set_label(COLORBAR_LABEL, fontsize=LABEL_FONTSIZE)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_png = args.out_dir / f"autoattack_eval_comparation_{args.plot_name}.png"
    out_svg = args.out_dir / f"autoattack_eval_comparation_{args.plot_name}.svg" if args.svg else None
    entries = model_entries(args.models_root, args.models, args.checkpoint_kind)
    data_by_norm = load_autoattack_results(args.models_root, entries, args.eval_eps_order)
    model_labels = [label for _model_name, label, _csv_name in entries]
    plot_comparation(data_by_norm, model_labels, args.eval_eps_order, args.plot_name, out_png, out_svg=out_svg, show=args.show)
    print(f"Saved AutoAttack comparation plot to {out_png}")
    if out_svg is not None:
        print(f"Saved AutoAttack comparation plot to {out_svg}")


if __name__ == "__main__":
    main()
