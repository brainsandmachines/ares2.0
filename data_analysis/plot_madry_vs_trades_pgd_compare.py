import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MADRY_CSV = "data_analysis/plots/madry/prepared_plot_data.csv"
DEFAULT_TRADES_CSV = "data_analysis/plots/trades/prepared_plot_data.csv"
DEFAULT_OUT_DIR = "data_analysis/plots_madry_vs_trades"
EVAL_NORMS = ["linf", "l2", "l1"]
TRAIN_NORMS = ["l2", "linf"]
TRAIN_EPS = [1.0, 2.0, 4.0, 8.0, 16.0]
EVAL_EPS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
INIT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
INIT_LINESTYLES = ["-", "--", "-.", ":"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Madry vs TRADES PGD comparison figures from prepared plot CSVs."
    )
    parser.add_argument("--madry-csv", default=DEFAULT_MADRY_CSV, help="Prepared madry plot CSV")
    parser.add_argument("--trades-csv", default=DEFAULT_TRADES_CSV, help="Prepared trades plot CSV")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=200, help="Saved figure DPI")
    return parser.parse_args()


def style_for_init(init_value: int) -> Tuple[str, str]:
    marker = INIT_MARKERS[(init_value - 1) % len(INIT_MARKERS)]
    linestyle = INIT_LINESTYLES[(init_value - 1) % len(INIT_LINESTYLES)]
    return marker, linestyle


def load_prepared_csv(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["method"] = method
    df["train_norm"] = df["train_family"].astype(str).str.lower()
    df["attack_norm"] = df["attack_norm"].astype(str).str.lower()
    df["train_eps"] = pd.to_numeric(df["train_eps"], errors="coerce")
    df["epsilon_input"] = pd.to_numeric(df["epsilon_input"], errors="coerce")
    df["clean_top1"] = pd.to_numeric(df["clean_top1"], errors="coerce")
    df["adv_top1"] = pd.to_numeric(df["adv_top1"], errors="coerce")
    df["init"] = pd.to_numeric(df["init"], errors="coerce").astype("Int64")
    df["model_dir"] = df["model_dir"].astype(str)
    return df


def load_all_data(madry_csv: Path, trades_csv: Path) -> pd.DataFrame:
    madry = load_prepared_csv(madry_csv, "madry")
    trades = load_prepared_csv(trades_csv, "trades")
    df = pd.concat([madry, trades], ignore_index=True)
    df = df[df["train_norm"].isin(TRAIN_NORMS)].copy()
    df = df[df["train_eps"].isin(TRAIN_EPS)].copy()
    df = df[df["epsilon_input"].isin(EVAL_EPS)].copy()
    return df


def model_label(model_dir: str) -> str:
    if model_dir.startswith("convnext_small_"):
        return model_dir.replace("convnext_small_", "", 1)
    return model_dir


def build_series(drun: pd.DataFrame, x_positions: Dict[float, int]) -> Tuple[List[int], List[float]]:
    dplot = drun.sort_values("epsilon_input")
    xs = [x_positions[0.0]]
    ys = [float(drun["clean_top1"].iloc[0])]
    for _, row in dplot.iterrows():
        eps = float(row["epsilon_input"])
        if eps in x_positions:
            xs.append(x_positions[eps])
            ys.append(float(row["adv_top1"]))
    return xs, ys


def plot_eval_figure(df: pd.DataFrame, eval_norm: str, out_dir: Path, dpi: int) -> Path:
    dnorm = df[df["attack_norm"] == eval_norm].copy()
    x_ticks = [0.0] + EVAL_EPS
    x_positions = {float(v): i for i, v in enumerate(x_ticks)}

    fig, axes = plt.subplots(2, 5, figsize=(26, 10), sharex=True, sharey=True, squeeze=False)

    for row_idx, train_norm in enumerate(TRAIN_NORMS):
        for col_idx, train_eps in enumerate(TRAIN_EPS):
            ax = axes[row_idx, col_idx]
            dpanel = dnorm[(dnorm["train_norm"] == train_norm) & (dnorm["train_eps"] == train_eps)].copy()
            ax.set_title(f"train {train_norm}, eps={train_eps:g}")

            if dpanel.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                run_order = (
                    dpanel[["model_dir", "init"]]
                    .drop_duplicates()
                    .sort_values(["model_dir", "init"], ascending=[True, True])
                    .reset_index(drop=True)
                )
                colors = plt.get_cmap("tab20")(range(max(len(run_order), 1)))
                for idx, run in run_order.iterrows():
                    drun = dpanel[dpanel["model_dir"] == run["model_dir"]].copy()
                    xs, ys = build_series(drun, x_positions)
                    marker, linestyle = style_for_init(int(run["init"]))
                    ax.plot(
                        xs,
                        ys,
                        color=colors[idx % len(colors)],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.8,
                        markersize=5,
                        alpha=0.95,
                        label=model_label(str(run["model_dir"])),
                    )
                ax.legend(fontsize=7, loc="best", frameon=True)

            ax.set_xticks(list(x_positions.values()))
            ax.set_xticklabels(["0"] + [f"{v:g}" for v in EVAL_EPS])
            ax.set_xlim(min(x_positions.values()), max(x_positions.values()))
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            if row_idx == 1:
                ax.set_xlabel("PGD eval constraint")
            if col_idx == 0:
                ax.set_ylabel("Top-1 Accuracy (%)")

    fig.suptitle(f"Madry vs TRADES, eval norm={eval_norm}", fontsize=16, y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pgd_compare_{eval_norm}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    df = load_all_data(Path(args.madry_csv), Path(args.trades_csv))
    out_dir = Path(args.out_dir)
    saved = [plot_eval_figure(df, eval_norm, out_dir, args.dpi) for eval_norm in EVAL_NORMS]
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
