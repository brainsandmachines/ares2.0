#!/usr/bin/env python3
"""Plot an AutoAttack L2 robustness heatmap for init1 L2 models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit these constants to change the figure without touching the data logic.
MODELS_ROOT = PROJECT_ROOT / "results" / "models"
OUTPUT_PATH = PROJECT_ROOT / "data_analysis" / "plots" / "autoattack_l2_models_heatmap.png"
SVG_OUTPUT_PATH = PROJECT_ROOT / "data_analysis" / "plots" / "autoattack_l2_models_heatmap.svg"
CSV_NAME = "autoattack_sweep_results.csv"
ATTACK_NORM = "l2"

TRAIN_EPS_ORDER = [0, 1, 2, 4, 8, 16]
EVAL_EPS_ORDER = [0, 1, 2, 4, 8, 16]
MODEL_DIR_BY_TRAIN_EPS = {
    0: "convnext_small_baseline_init1",
    1: "convnext_small_l2_1_init1",
    2: "convnext_small_l2_2_init1",
    4: "convnext_small_l2_4_init6",
    8: "convnext_small_l2_8_init3",
    16: "convnext_small_l2_16_init3",
}

TITLE = "AutoAttack robustness - $\ell_2$ models"
X_LABEL = r"Model trained $\ell_2$ epsilon"
Y_LABEL = r"Evaluated $\ell_2$ AutoAttack robustness"
COLORBAR_LABEL = "Accuracy (%)"
CMAP = "plasma"
VMIN = 0.0
VMAX = 100.0
FIGSIZE = (4.3, 4)
DPI = 200
ANNOTATION_FMT = "{:.1f}%"
ANNOTATION_FONTSIZE = 10
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 13
TEXT_DARK_THRESHOLD = 20.0
GRID_COLOR = "white"
GRID_LINEWIDTH = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AutoAttack L2 robustness heatmap for init1 L2 models.")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT, help="Root directory containing model folders.")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH, help="Output PNG path.")
    parser.add_argument("--svg-out", type=Path, default=SVG_OUTPUT_PATH, help="Output SVG path.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving.")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, csv_path: Path, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")


def accuracy_for_eval_eps(df_l2: pd.DataFrame, csv_path: Path, eval_eps: int) -> float:
    if eval_eps == 0:
        clean_values = pd.to_numeric(df_l2["clean_acc"], errors="coerce").dropna().unique()
        if len(clean_values) == 0:
            raise ValueError(f"{csv_path} has no clean_acc values for {ATTACK_NORM} rows.")
        return float(np.mean(clean_values))

    rows = df_l2[np.isclose(pd.to_numeric(df_l2["epsilon_input"], errors="coerce"), float(eval_eps))]
    if rows.empty:
        raise ValueError(f"{csv_path} has no {ATTACK_NORM} row with epsilon_input={eval_eps}.")
    robust_values = pd.to_numeric(rows["robust_acc"], errors="coerce").dropna()
    if robust_values.empty:
        raise ValueError(f"{csv_path} has no robust_acc value for epsilon_input={eval_eps}.")
    return float(robust_values.mean())


def build_heatmap(models_root: Path) -> np.ndarray:
    data = np.full((len(EVAL_EPS_ORDER), len(TRAIN_EPS_ORDER)), np.nan, dtype=float)

    for col_idx, train_eps in enumerate(TRAIN_EPS_ORDER):
        model_dir = MODEL_DIR_BY_TRAIN_EPS[train_eps]
        csv_path = models_root / model_dir / CSV_NAME
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing AutoAttack sweep CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        require_columns(df, csv_path, ["attack_norm", "epsilon_input", "clean_acc", "robust_acc"])
        df_l2 = df[df["attack_norm"].astype(str).str.lower() == ATTACK_NORM].copy()
        if df_l2.empty:
            raise ValueError(f"{csv_path} has no rows with attack_norm={ATTACK_NORM!r}.")

        for row_idx, eval_eps in enumerate(EVAL_EPS_ORDER):
            data[row_idx, col_idx] = accuracy_for_eval_eps(df_l2, csv_path, eval_eps)

    return data


def render_heatmap(data: np.ndarray, out_path: Path, svg_out_path: Path, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    im = ax.imshow(data, origin="lower", aspect="equal", cmap=CMAP, vmin=VMIN, vmax=VMAX)

    ax.set_xticks(range(len(TRAIN_EPS_ORDER)))
    ax.set_yticks(range(len(EVAL_EPS_ORDER)))
    ax.set_xticklabels([str(eps) for eps in TRAIN_EPS_ORDER], fontsize=TICK_FONTSIZE)
    ax.set_yticklabels([str(eps) for eps in EVAL_EPS_ORDER], fontsize=TICK_FONTSIZE)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONTSIZE)
    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE,x=0.45)

    ax.set_xticks(np.arange(-0.5, len(TRAIN_EPS_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(EVAL_EPS_ORDER), 1), minor=True)
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

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
    cbar.set_label(COLORBAR_LABEL, fontsize=LABEL_FONTSIZE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    svg_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    fig.savefig(svg_out_path)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = build_heatmap(args.models_root)
    render_heatmap(data, args.out, args.svg_out, show=args.show)
    print(f"Saved AutoAttack L2 heatmap to {args.out}")
    print(f"Saved AutoAttack L2 heatmap to {args.svg_out}")


if __name__ == "__main__":
    main()
