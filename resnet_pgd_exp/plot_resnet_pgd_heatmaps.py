import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NORM_ORDER = ["linf", "l2", "l1"]
LINF_DIVISOR = 255.0
L1_MULTIPLIER = 255.0 / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-norm PGD heatmaps (x=model eps, y=eval eps) for Madry ResNet50 runs"
    )
    parser.add_argument(
        "--results-root",
        default="resnet_pgd_exp/results",
        help="Root directory that contains per-model folders with pgd_validation_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="resnet_pgd_exp/plots/madry/norm_matrix",
        help="Output directory for heatmaps",
    )
    parser.add_argument(
        "--metric-col",
        default="adv_top1",
        choices=["adv_top1", "adv_top5", "clean_top1", "clean_top5"],
        help="Metric to render in heatmap cells",
    )
    return parser.parse_args()


def _format_tick_values(vals: List[float]) -> List[str]:
    labels = []
    for v in vals:
        if pd.isna(v):
            labels.append("nan")
        elif abs(v - round(v)) < 1e-12:
            labels.append(str(int(round(v))))
        elif abs(v) < 0.1:
            labels.append(f"{v:.3f}".rstrip("0").rstrip("."))
        else:
            labels.append(f"{v:g}")
    return labels


def _compute_pgd_constrained_eps(eps_input: float, attack_norm: str) -> float:
    n = str(attack_norm).lower()
    e = float(eps_input)
    if n == "linf":
        return e / LINF_DIVISOR
    if n == "l1":
        return e * L1_MULTIPLIER
    return e


def _tick_label_for_norm(x: float, norm: str) -> str:
    if abs(float(x)) < 1e-12:
        return "0"
    if str(norm).lower() == "linf":
        num = float(x) * LINF_DIVISOR
        rounded = round(num)
        if abs(num - rounded) < 1e-8:
            return f"{int(rounded)}/255"
        return f"{num:g}/255"
    return f"{float(x):g}"


def _format_tick_values_by_norm(vals: List[float], norm: str) -> List[str]:
    return [_tick_label_for_norm(float(v), norm) for v in vals]


def _render_heatmap(ax, data: np.ndarray, xlabels: List[str], ylabels: List[str]):
    im = ax.imshow(data, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=100)

    ax.set_xticks(list(range(len(xlabels))))
    ax.set_yticks(list(range(len(ylabels))))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = float(data[i, j])
            if np.isnan(val):
                continue
            txt_color = "white" if val < 20.0 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=8)
    return im


def _load_results(results_root: Path) -> pd.DataFrame:
    csv_paths = sorted(results_root.rglob("pgd_validation_results.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No pgd_validation_results.csv files found under {results_root}")

    frames = []
    for p in csv_paths:
        d = pd.read_csv(p)
        d["_source_csv"] = str(p)
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    required = {"attack_norm", "train_eps", "epsilon_input", "adv_top1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["attack_norm"] = df["attack_norm"].astype(str).str.lower()
    df["train_eps"] = pd.to_numeric(df["train_eps"], errors="coerce")
    df["epsilon_input"] = pd.to_numeric(df["epsilon_input"], errors="coerce")
    for col in ["adv_top1", "adv_top5", "clean_top1", "clean_top5"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_norm_heatmap(df: pd.DataFrame, norm: str, metric_col: str, out_dir: Path) -> str:
    d = df[df["attack_norm"] == norm].copy()
    if d.empty:
        raise RuntimeError(f"No rows found for attack_norm={norm}")

    if "train_norm" in d.columns:
        d["train_eps_plot"] = [
            _compute_pgd_constrained_eps(eps_input=e, attack_norm=tnorm) if pd.notna(e) else float("nan")
            for e, tnorm in zip(d["train_eps"], d["train_norm"])
        ]
        train_norm_for_ticks = str(d["train_norm"].dropna().astype(str).str.lower().iloc[0]) if not d["train_norm"].dropna().empty else "l2"
    else:
        d["train_eps_plot"] = d["train_eps"]
        train_norm_for_ticks = "l2"

    d["eval_eps_plot"] = [
        _compute_pgd_constrained_eps(eps_input=e, attack_norm=norm) if pd.notna(e) else float("nan")
        for e in d["epsilon_input"]
    ]

    pivot = d.pivot_table(
        index="eval_eps_plot",
        columns="train_eps_plot",
        values=metric_col,
        aggfunc="mean",
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    # Keep only one clean representation: the explicit "clean" row.
    if not pivot.empty:
        pivot = pivot.loc[np.abs(pivot.index.to_numpy(dtype=float)) > 1e-12]
    if pivot.empty:
        raise RuntimeError(f"Pivot became empty for norm={norm}")

    clean_metric_col = metric_col.replace("adv_", "clean_") if metric_col.startswith("adv_") else metric_col
    clean_by_train = (
        d.groupby("train_eps_plot", dropna=True)[clean_metric_col]
        .mean()
        .reindex(pivot.columns)
    )

    data = np.vstack([clean_by_train.to_numpy(dtype=float), pivot.to_numpy(dtype=float)])
    xvals = [float(v) for v in pivot.columns.tolist()]
    yvals = [float(v) for v in pivot.index.tolist()]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = _render_heatmap(
        ax=ax,
        data=data,
        xlabels=_format_tick_values_by_norm(xvals, train_norm_for_ticks),
        ylabels=["clean"] + _format_tick_values_by_norm(yvals, norm),
    )
    ax.set_title(f"Madry ResNet50 PGD Heatmap ({norm})")
    ax.set_xlabel("Trained constrained eps")
    ax.set_ylabel("Evaluated constrained eps")

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Top-1 Accuracy (%)" if metric_col == "adv_top1" else f"{metric_col} (%)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pgd_train_eval_eps_heatmap_{norm}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)

    df = _load_results(results_root)

    saved = []
    for norm in NORM_ORDER:
        if norm not in set(df["attack_norm"].unique().tolist()):
            continue
        saved.append(save_norm_heatmap(df, norm, args.metric_col, out_dir))

    if not saved:
        raise RuntimeError("No heatmaps were produced. Check available attack_norm values in CSV files.")

    print(f"Saved {len(saved)} heatmap(s):")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
