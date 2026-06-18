#!/usr/bin/env python3
"""Plot AutoAttack sweep results for the external GradNorm models."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "results" / "rig_gradnorm_resnet50_gelu" / "autoattack_sweep_results.csv"
DEFAULT_COMPARE_CSV = Path(
    "/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models/"
    "convnext_small_l2_1_init1/autoattack_sweep_results.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--compare-csv", type=Path, default=DEFAULT_COMPARE_CSV)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def pct(value: str) -> float:
    raw = float(value)
    return raw * 100.0 if raw <= 1.000001 else raw


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", newline="")))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def model_label(rows: list[dict[str, str]], fallback: str) -> str:
    first = rows[0]
    return first.get("display_name") or first.get("model_name") or fallback


def group_results(rows: list[dict[str, str]]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row["attack_norm"] not in {"linf", "l2"}:
            continue
        grouped[row["attack_norm"]].append((float(row["epsilon_input"]), pct(row["robust_acc"])))
    return grouped


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.csv.parent / "plots"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv)
    compare_rows = load_rows(args.compare_csv) if args.compare_csv.exists() else []

    model_name = model_label(rows, args.csv.parent.name)
    clean_acc = pct(rows[0]["clean_acc"])
    num_images = rows[0]["num_images"]
    grouped = group_results(rows)
    compare_grouped = group_results(compare_rows) if compare_rows else {}
    compare_name = model_label(compare_rows, args.compare_csv.parent.name) if compare_rows else None
    compare_clean = pct(compare_rows[0]["clean_acc"]) if compare_rows else None
    compare_num_images = compare_rows[0].get("num_images") if compare_rows else None

    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    styles = {
        ("rig", "linf"): {"label": f"{model_name} Linf (eps/255)", "marker": "o", "color": "#2a6fbb", "linestyle": "-"},
        ("rig", "l2"): {"label": f"{model_name} L2", "marker": "s", "color": "#b4482c", "linestyle": "-"},
        ("compare", "linf"): {"label": f"{compare_name} Linf (eps/255)" if compare_name else "Compare Linf", "marker": "o", "color": "#6aa84f", "linestyle": "--"},
        ("compare", "l2"): {"label": f"{compare_name} L2" if compare_name else "Compare L2", "marker": "s", "color": "#8e5ea2", "linestyle": "--"},
    }
    for norm in ("linf", "l2"):
        values = sorted(grouped.get(norm, []))
        if not values:
            continue
        xs = [x for x, _ in values]
        ys = [y for _, y in values]
        ax.plot(xs, ys, linewidth=2.0, markersize=6, **styles[("rig", norm)])
    for norm in ("linf", "l2"):
        values = sorted(compare_grouped.get(norm, []))
        if not values:
            continue
        xs = [x for x, _ in values]
        ys = [y for _, y in values]
        ax.plot(xs, ys, linewidth=2.0, markersize=6, **styles[("compare", norm)])

    ax.axhline(clean_acc, color="#333333", linestyle="--", linewidth=1.25, label=f"Clean {clean_acc:.2f}%")
    if compare_clean is not None:
        ax.axhline(compare_clean, color="#777777", linestyle=":", linewidth=1.25, label=f"{compare_name} clean {compare_clean:.2f}%")
    title = f"AutoAttack Sweep: {model_name}"
    if compare_name:
        title += f" vs {compare_name}"
    ax.set_title(title)
    ax.set_xlabel("Epsilon input")
    ax.set_ylabel("Accuracy (%)")
    xticks = {float(row["epsilon_input"]) for row in rows}
    xticks.update(float(row["epsilon_input"]) for row in compare_rows if row["attack_norm"] in {"linf", "l2"})
    ax.set_xticks(sorted(xticks))
    y_max = max(65, clean_acc + 5, (compare_clean or 0) + 5)
    ax.set_ylim(0, y_max)
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.01,
        0.02,
        f"RIG n={num_images}" + (f"; compare n={compare_num_images}" if compare_num_images else ""),
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )

    stem = args.csv.stem.replace("_results", "")
    suffix = "_with_l2_1_init1" if compare_rows else ""
    png = args.out_dir / f"{stem}_plot{suffix}.png"
    svg = args.out_dir / f"{stem}_plot{suffix}.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
