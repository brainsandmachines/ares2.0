import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------
# Cluster-editable paths
# ---------------------------
DEFAULT_CSV = "data_analysis/pgd_validation_results.csv"
DEFAULT_OUT_DIR = "data_analysis/plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PGD sweep accuracy vs epsilon")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV from run_pgd_validation_sweep.py")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output plot root directory")
    parser.add_argument(
        "--x-col",
        default="epsilon_input",
        choices=["epsilon_eval", "epsilon_input"],
        help="X-axis epsilon column",
    )
    parser.add_argument("--make-norm-plots", action="store_true", help="Also save single-norm plots")
    return parser.parse_args()


def _norm_order(norms):
    preferred = ["linf", "l2", "l1"]
    ordered = [n for n in preferred if n in norms]
    ordered.extend([n for n in sorted(norms) if n not in ordered])
    return ordered


def save_combined_plot(df_model: pd.DataFrame, model_name: str, category: str, init: str, out_dir: str, x_col: str) -> str:
    out_path = Path(out_dir) / category / "combined" / f"init{init}" / f"{model_name}_combined.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for norm in _norm_order(df_model["attack_norm"].unique()):
        d = df_model[df_model["attack_norm"] == norm].sort_values(x_col)
        if d.empty:
            continue
        plt.plot(d[x_col], d["adv_top1"], marker="o", label=norm)

    plt.xlabel(x_col)
    plt.ylabel("Adv Acc@1 (%)")
    plt.title(f"{model_name}: PGD robustness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


def save_norm_plot(df_model: pd.DataFrame, model_name: str, category: str, init: str, norm: str, out_dir: str, x_col: str) -> str:
    out_path = Path(out_dir) / category / norm / f"init{init}" / f"{model_name}_{norm}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = df_model[df_model["attack_norm"] == norm].sort_values(x_col)
    if d.empty:
        return ""

    plt.figure(figsize=(8, 5))
    plt.plot(d[x_col], d["adv_top1"], marker="o", label=norm)
    plt.xlabel(x_col)
    plt.ylabel("Adv Acc@1 (%)")
    plt.title(f"{model_name}: {norm} PGD")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    req_cols = {"model_name", "category", "init", "attack_norm", "adv_top1", args.x_col}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    saved = []
    for model_name, dmodel in df.groupby("model_name"):
        category = str(dmodel["category"].iloc[0])
        init = str(dmodel["init"].iloc[0])

        saved.append(save_combined_plot(dmodel, model_name, category, init, args.out_dir, args.x_col))

        if args.make_norm_plots:
            for norm in _norm_order(dmodel["attack_norm"].unique()):
                p = save_norm_plot(dmodel, model_name, category, init, norm, args.out_dir, args.x_col)
                if p:
                    saved.append(p)

    print(f"Saved {len(saved)} plot(s).")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
