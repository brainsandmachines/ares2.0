import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


KEY_METHOD_ORDER = [
    "baseline_l2norm",
    "l1norm_step",
    "raw_grad_step",
    "l1_apgd_topk_rho0.01",
    "l1_apgd_topk_rho0.05",
    "l1_apgd_topk_rho0.10",
    "fw_onehot",
]

LOSS_METHODS = [
    "baseline_l2norm",
    "l1_apgd_topk_rho0.01",
    "l1_apgd_topk_rho0.05",
    "fw_onehot",
]


def method_tag(row: pd.Series) -> str:
    method = str(row["method"])
    rho = row.get("rho", "")
    if pd.isna(rho) or str(rho).strip() == "":
        return method
    return f"{method}_rho{float(rho):.2f}"


def short_model_name(model_name: str) -> str:
    return str(model_name).split("/")[-1]


def make_line_plot(df: pd.DataFrame, model_name: str, y_col: str, y_label: str, out_path: Path) -> None:
    d = df[df["model_name"] == model_name].copy()
    if d.empty:
        return

    d["method_tag"] = d.apply(method_tag, axis=1)
    plt.figure(figsize=(9, 6))

    methods = [m for m in KEY_METHOD_ORDER if m in set(d["method_tag"])]
    for m in methods:
        dm = d[d["method_tag"] == m].sort_values("epsilon_input")
        plt.plot(dm["epsilon_input"], dm[y_col], marker="o", linewidth=2, label=m)

    plt.title(f"{y_label} vs Epsilon\n{model_name}")
    plt.xlabel("epsilon_input")
    plt.ylabel(y_label)
    plt.xticks(sorted(d["epsilon_input"].unique()))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_loss_progression_plots(loss_df: pd.DataFrame, out_dir: Path) -> None:
    if loss_df.empty:
        return

    loss_df = loss_df.copy()
    loss_df["method_tag"] = loss_df.apply(method_tag, axis=1)

    for model_name in sorted(loss_df["model_name"].unique()):
        dm = loss_df[loss_df["model_name"] == model_name]
        for eps in sorted(dm["epsilon_input"].unique()):
            de = dm[dm["epsilon_input"] == eps]
            if de.empty:
                continue

            plt.figure(figsize=(8, 5))
            for method in LOSS_METHODS:
                dme = de[de["method_tag"] == method]
                if dme.empty:
                    continue
                davg = dme.groupby("iteration", as_index=False)["mean_loss"].mean().sort_values("iteration")
                plt.plot(davg["iteration"], davg["mean_loss"], marker="o", linewidth=2, label=method)

            plt.title(f"Loss Progression (mean over runs)\n{model_name} | eps={eps:g}")
            plt.xlabel("Iteration")
            plt.ylabel("Mean Loss")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=8)
            out_path = out_dir / f"loss_progression_{model_name.replace('/', '_')}_eps{eps:g}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, dpi=160)
            plt.close()


def _method_pivot(agg_df: pd.DataFrame, method: str, model_order, eps_order) -> pd.DataFrame:
    dm = agg_df[agg_df["method_tag"] == method]
    return (
        dm.pivot_table(
            index="model_name",
            columns="epsilon_input",
            values="success_rate",
            aggfunc="mean",
        )
        .reindex(index=model_order, columns=eps_order)
    )


def make_success_heatmaps_per_method(agg_df: pd.DataFrame, out_dir: Path) -> None:
    d = agg_df.copy()
    d["method_tag"] = d.apply(method_tag, axis=1)

    model_order = sorted(d["model_name"].unique())
    eps_order = sorted(d["epsilon_input"].unique())

    for method in [m for m in KEY_METHOD_ORDER if m in set(d["method_tag"])]:
        pivot = _method_pivot(d, method, model_order, eps_order)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        img = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)

        ax.set_title(f"Success Rate Heatmap | {method}")
        ax.set_xlabel("epsilon_input")
        ax.set_ylabel("model")
        ax.set_xticks(range(len(eps_order)))
        ax.set_xticklabels([f"{e:g}" for e in eps_order])
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels([short_model_name(m) for m in model_order])

        for i in range(len(model_order)):
            for j in range(len(eps_order)):
                v = pivot.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white", fontsize=8)

        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("success_rate (%)")

        out_path = out_dir / f"heatmap_success_{method}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()


def make_success_heatmap_grid(agg_df: pd.DataFrame, out_dir: Path) -> None:
    d = agg_df.copy()
    d["method_tag"] = d.apply(method_tag, axis=1)

    methods = [m for m in KEY_METHOD_ORDER if m in set(d["method_tag"])]
    if not methods:
        return

    model_order = sorted(d["model_name"].unique())
    eps_order = sorted(d["epsilon_input"].unique())

    n = len(methods)
    ncols = 4
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False)
    cmap = "viridis"
    vmin, vmax = 0.0, 100.0
    img = None

    for idx, method in enumerate(methods):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        pivot = _method_pivot(d, method, model_order, eps_order)
        img = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(method, fontsize=9)
        ax.set_xticks(range(len(eps_order)))
        ax.set_xticklabels([f"{e:g}" for e in eps_order], fontsize=8)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels([short_model_name(m) for m in model_order], fontsize=8)
        ax.set_xlabel("eps", fontsize=8)

        if c == 0:
            ax.set_ylabel("model", fontsize=8)
        else:
            ax.set_yticklabels([])

        for i in range(len(model_order)):
            for j in range(len(eps_order)):
                v = pivot.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white", fontsize=7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Success Rate Heatmaps by Method", fontsize=12)

    # Reserve space for a dedicated right-side colorbar axis.
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.96])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label("success_rate (%)")

    out_path = out_dir / "heatmap_success_all_methods.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot L1 step-strategy benchmark results")
    parser.add_argument("--input-dir", default="data_analysis/l1_step_strategy_benchmark")
    parser.add_argument("--output-dir", default="data_analysis/l1_step_strategy_benchmark/plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = input_dir / "results_agg.csv"
    loss_path = input_dir / "loss_traces.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path}")
    if not loss_path.exists():
        raise FileNotFoundError(f"Missing {loss_path}")

    agg = pd.read_csv(agg_path)
    loss = pd.read_csv(loss_path)

    for model_name in sorted(agg["model_name"].unique()):
        make_line_plot(
            agg,
            model_name,
            y_col="success_rate",
            y_label="Attack Success Rate (%)",
            out_path=out_dir / f"success_vs_eps_{model_name.replace('/', '_')}.png",
        )
        make_line_plot(
            agg,
            model_name,
            y_col="mean_l0",
            y_label="Sparsity (mean L0)",
            out_path=out_dir / f"sparsity_vs_eps_{model_name.replace('/', '_')}.png",
        )
        make_line_plot(
            agg,
            model_name,
            y_col="efficiency",
            y_label="Efficiency (success rate / sec per image)",
            out_path=out_dir / f"efficiency_vs_eps_{model_name.replace('/', '_')}.png",
        )

    make_loss_progression_plots(loss, out_dir=out_dir)
    make_success_heatmaps_per_method(agg, out_dir=out_dir)
    make_success_heatmap_grid(agg, out_dir=out_dir)


if __name__ == "__main__":
    main()
