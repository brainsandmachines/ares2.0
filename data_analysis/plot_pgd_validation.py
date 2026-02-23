import argparse
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV = "data_analysis/pgd_validation_results.csv"
DEFAULT_OUT_DIR = "data_analysis/plots"
DEFAULT_MODELS_DIR = "/home/ashtomer/projects/ares/results/models"
NORM_ORDER = ["linf", "l2", "l1"]
EXCLUDED_NON_MADRY = ("gradnorm", "trades", "baseline")
LINF_DIVISOR = 255.0
L1_MULTIPLIER = 255.0 / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PGD sweep accuracy vs epsilon")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV (legacy/single-file mode)")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="Root dir for auto-discovered model PGD CSVs")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output plot root directory")
    parser.add_argument(
        "--x-col",
        default="pgd_constrained_eps",
        choices=["pgd_constrained_eps", "epsilon_eval", "epsilon_input"],
        help="X-axis epsilon column",
    )
    parser.add_argument(
        "--auto-collect-pgd-csvs",
        action="store_true",
        default=True,
        help="Auto-collect */pgd_eval/pgd_validation_results.csv from --models-dir",
    )
    parser.add_argument(
        "--no-auto-collect-pgd-csvs",
        action="store_false",
        dest="auto_collect_pgd_csvs",
        help="Disable auto-collect and use --csv",
    )
    parser.add_argument(
        "--filter-madry-only",
        action="store_true",
        default=True,
        help="Keep only madry-like model dirs (exclude gradnorm/trades/baseline)",
    )
    parser.add_argument(
        "--no-filter-madry-only",
        action="store_false",
        dest="filter_madry_only",
        help="Disable madry-only folder filtering",
    )
    parser.add_argument("--make-matrix-plot", action="store_true", default=True, help="Save a 3x3 train/eval norm matrix")
    parser.add_argument(
        "--no-make-matrix-plot",
        action="store_false",
        dest="make_matrix_plot",
        help="Disable the 3x3 matrix plot",
    )
    parser.add_argument("--make-pair-plots", action="store_true", default=True, help="Save per-(train_norm,attack_norm) plots")
    parser.add_argument(
        "--no-make-pair-plots",
        action="store_false",
        dest="make_pair_plots",
        help="Disable per-pair plots",
    )
    return parser.parse_args()


def _norm_order(norms: List[str]) -> List[str]:
    ordered = [n for n in NORM_ORDER if n in norms]
    ordered.extend([n for n in sorted(norms) if n not in ordered])
    return ordered


def _is_madry_dir(name: str) -> bool:
    low = name.lower()
    return not any(t in low for t in EXCLUDED_NON_MADRY)


def _trim_float(v: Optional[float]) -> str:
    if v is None:
        return "unknown"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "unknown"
    if fv.is_integer():
        return str(int(fv))
    return f"{fv:g}"


def _compute_pgd_constrained_eps(eps_input: float, attack_norm: str) -> float:
    n = str(attack_norm).lower()
    e = float(eps_input)
    if n == "linf":
        return e / LINF_DIVISOR
    if n == "l1":
        return e * L1_MULTIPLIER
    return e


def _tick_label_for_norm(x: float, attack_norm: str) -> str:
    if abs(float(x)) < 1e-12:
        return "0"
    if str(attack_norm).lower() == "linf":
        num = float(x) * LINF_DIVISOR
        rounded = round(num)
        if abs(num - rounded) < 1e-8:
            return f"{int(rounded)}/255"
        return f"{_trim_float(num)}/255"
    return _trim_float(float(x))


def _apply_x_scaling_and_ticks(ax, xticks: List[float], attack_norm: str) -> None:
    pos = sorted([t for t in xticks if t > 0.0])
    if pos:
        # Keep eps=0 visible while using logarithmic scaling for positive eps.
        linthresh = pos[0] / 2.0
        ax.set_xscale("symlog", linthresh=linthresh, base=2)

    if xticks:
        ax.set_xticks(xticks)
        labels = [_tick_label_for_norm(t, attack_norm) for t in xticks]
        ax.set_xticklabels(labels)

        xmin, xmax = min(xticks), max(xticks)
        if xmin == xmax:
            ax.set_xlim(max(0.0, xmin - 0.5), xmax + 0.5)
        else:
            pad = 0.03 * (xmax - xmin)
            ax.set_xlim(max(0.0, xmin), xmax + pad)


def _parse_meta_from_model_dir(model_dir_name: str) -> Dict[str, str]:
    low = (model_dir_name or "").lower()
    train_norm = "unknown"
    m_norm = re.search(r"(^|[_\-])(linf|l2|l1)($|[_\-])", low)
    if m_norm:
        train_norm = m_norm.group(2)

    init = "unknown"
    m_init = re.search(r"init[_\-]?(\d+)", low)
    if m_init:
        init = m_init.group(1)

    train_eps = None
    m_eps = re.search(r"(?:linf|l2|l1)[_\-]?([0-9]*\.?[0-9]+)", low)
    if m_eps:
        try:
            train_eps = float(m_eps.group(1))
        except ValueError:
            train_eps = None

    return {
        "train_norm": train_norm,
        "init": init,
        "train_eps": train_eps,
    }


def _extract_model_dir_from_row(row: pd.Series) -> str:
    cp = row.get("checkpoint_path", "")
    if isinstance(cp, str) and cp:
        try:
            return Path(cp).parent.name
        except Exception:
            pass

    model_dir_name = row.get("_model_dir_name", "")
    if isinstance(model_dir_name, str) and model_dir_name:
        return model_dir_name

    model_name = row.get("model_name", "")
    if isinstance(model_name, str) and model_name:
        return model_name

    return "unknown_model"


def _discover_csvs(models_dir: str, filter_madry_only: bool) -> List[Path]:
    root = Path(models_dir)
    csvs = sorted(root.glob("*/pgd_eval/pgd_validation_results.csv"))
    if filter_madry_only:
        csvs = [p for p in csvs if _is_madry_dir(p.parent.parent.name)]
    return csvs


def _load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.auto_collect_pgd_csvs:
        csv_paths = _discover_csvs(args.models_dir, args.filter_madry_only)
        if not csv_paths:
            raise RuntimeError(f"No PGD CSV files found under {args.models_dir}")

        frames = []
        for p in csv_paths:
            d = pd.read_csv(p)
            if d.empty:
                continue
            d["_source_csv"] = str(p)
            d["_model_dir_name"] = p.parent.parent.name
            frames.append(d)

        if not frames:
            raise RuntimeError("All discovered PGD CSV files were empty.")

        return pd.concat(frames, ignore_index=True)

    return pd.read_csv(args.csv)


def _enrich_metadata(df: pd.DataFrame, filter_madry_only: bool) -> pd.DataFrame:
    df = df.copy()

    model_dirs = [_extract_model_dir_from_row(row) for _, row in df.iterrows()]
    df["model_dir"] = model_dirs

    parsed = [_parse_meta_from_model_dir(m) for m in model_dirs]
    df["_parsed_train_norm"] = [p["train_norm"] for p in parsed]
    df["_parsed_init"] = [p["init"] for p in parsed]
    df["train_eps"] = [p["train_eps"] for p in parsed]

    if "train_norm" not in df.columns:
        df["train_norm"] = "unknown"
    if "init" not in df.columns:
        df["init"] = "unknown"
    if "category" not in df.columns:
        df["category"] = "unknown"

    bad_train_norm = ~df["train_norm"].astype(str).str.lower().isin(NORM_ORDER)
    df.loc[bad_train_norm, "train_norm"] = df.loc[bad_train_norm, "_parsed_train_norm"]

    bad_init = df["init"].astype(str).str.lower().isin(["unknown", "nan", "none", ""])
    df.loc[bad_init, "init"] = df.loc[bad_init, "_parsed_init"]

    if filter_madry_only:
        df["category"] = "madry"

    if "attack_norm" not in df.columns:
        raise ValueError("Missing required column: attack_norm")
    df["attack_norm"] = df["attack_norm"].astype(str).str.lower()

    if "epsilon_input" not in df.columns:
        raise ValueError("Missing required column: epsilon_input")
    df["epsilon_input"] = pd.to_numeric(df["epsilon_input"], errors="coerce")
    df["pgd_constrained_eps"] = [
        _compute_pgd_constrained_eps(eps_input=e, attack_norm=n) if pd.notna(e) else float("nan")
        for e, n in zip(df["epsilon_input"], df["attack_norm"])
    ]

    df["model_id"] = df["model_dir"].astype(str)
    labels = []
    for eps, init in zip(df["train_eps"], df["init"]):
        init_str = str(init)
        if init_str.lower() in ("unknown", "nan", "none", ""):
            labels.append(f"eps={_trim_float(eps)}")
        else:
            labels.append(f"eps={_trim_float(eps)} init {init_str}")
    df["model_label"] = labels
    return df


def _build_norm_ticks(df: pd.DataFrame, x_col: str) -> Dict[str, List[float]]:
    ticks: Dict[str, List[float]] = {}
    for norm in NORM_ORDER:
        dn = df[df["attack_norm"] == norm]
        vals = []
        if x_col in dn.columns:
            vals = [float(v) for v in dn[x_col].dropna().unique().tolist()]
        vals.append(0.0)
        vals = sorted(set(vals))
        ticks[norm] = vals
    return ticks


def _plot_pair_subplot(ax, dpair: pd.DataFrame, train_norm: str, attack_norm: str, x_col: str, xticks: List[float]) -> None:
    ax.set_title(f"train={train_norm}, eval={attack_norm}")

    if dpair.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _apply_x_scaling_and_ticks(ax, xticks, attack_norm)
        ax.grid(True, alpha=0.3)
        return

    model_order = (
        dpair[["model_id", "train_eps", "init"]]
        .drop_duplicates(subset=["model_id"])
        .assign(
            _eps_sort=lambda t: pd.to_numeric(t["train_eps"], errors="coerce"),
            _init_sort=lambda t: pd.to_numeric(t["init"], errors="coerce"),
        )
        .sort_values(by=["_eps_sort", "_init_sort", "model_id"], ascending=[True, True, True], na_position="last")
    )

    for model_id in model_order["model_id"].tolist():
        dmodel = dpair[dpair["model_id"] == model_id]
        dmodel = dmodel.sort_values(x_col)
        robust_x = [float(v) for v in dmodel[x_col].tolist()]
        robust_y = [float(v) for v in dmodel["adv_top1"].tolist()]

        clean_top1 = float(dmodel["clean_top1"].iloc[0]) if "clean_top1" in dmodel.columns else None
        x_vals = robust_x[:]
        y_vals = robust_y[:]

        if clean_top1 is not None and 0.0 not in x_vals:
            x_vals.append(0.0)
            y_vals.append(clean_top1)

        xy = sorted(zip(x_vals, y_vals), key=lambda t: t[0])
        x_sorted = [t[0] for t in xy]
        y_sorted = [t[1] for t in xy]

        label = str(dmodel["model_label"].iloc[0])
        ax.plot(x_sorted, y_sorted, marker="o", label=label)

    _apply_x_scaling_and_ticks(ax, xticks, attack_norm)

    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def save_norm_matrix_plot(df: pd.DataFrame, out_dir: str, x_col: str, madry_only: bool) -> str:
    sub = "madry" if madry_only else "all"
    out_path = Path(out_dir) / sub / "norm_matrix" / "pgd_train_vs_eval_3x3.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    norm_ticks = _build_norm_ticks(df, x_col)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharey=True)
    for row, attack_norm in enumerate(NORM_ORDER):
        for col, train_norm in enumerate(NORM_ORDER):
            ax = axes[row, col]
            dpair = df[(df["train_norm"] == train_norm) & (df["attack_norm"] == attack_norm)]
            _plot_pair_subplot(ax, dpair, train_norm, attack_norm, x_col, norm_ticks.get(attack_norm, []))

            if row == len(NORM_ORDER) - 1:
                ax.set_xlabel("PGD constrained eps" if x_col == "pgd_constrained_eps" else x_col)
            if col == 0:
                ax.set_ylabel("Top-1 Accuracy (%)")

    fig.suptitle("PGD Robustness Matrix (rows=eval norm, cols=train norm)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def save_pair_plot(df: pd.DataFrame, out_dir: str, x_col: str, train_norm: str, attack_norm: str, madry_only: bool) -> str:
    sub = "madry" if madry_only else "all"
    out_path = Path(out_dir) / sub / "pairs" / f"train_{train_norm}__eval_{attack_norm}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpair = df[(df["train_norm"] == train_norm) & (df["attack_norm"] == attack_norm)]
    ticks = _build_norm_ticks(df, x_col).get(attack_norm, [])

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_pair_subplot(ax, dpair, train_norm, attack_norm, x_col, ticks)
    ax.set_xlabel("PGD constrained eps" if x_col == "pgd_constrained_eps" else x_col)
    ax.set_ylabel("Top-1 Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


# Legacy compatibility for final_eval imports.
def save_combined_plot(df_model: pd.DataFrame, model_name: str, category: str, init: str, out_dir: str, x_col: str) -> str:
    out_path = Path(out_dir) / category / "combined" / f"init{init}" / f"{model_name}_combined.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for norm in _norm_order(df_model["attack_norm"].astype(str).str.lower().unique().tolist()):
        d = df_model[df_model["attack_norm"].astype(str).str.lower() == norm].sort_values(x_col)
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


# Legacy compatibility for final_eval imports.
def save_norm_plot(df_model: pd.DataFrame, model_name: str, category: str, init: str, norm: str, out_dir: str, x_col: str) -> str:
    out_path = Path(out_dir) / category / norm / f"init{init}" / f"{model_name}_{norm}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = df_model.copy()
    d["attack_norm"] = d["attack_norm"].astype(str).str.lower()
    norm = str(norm).lower()
    d = d[d["attack_norm"] == norm]
    if d.empty:
        return ""

    if "epsilon_input" not in d.columns:
        raise ValueError("save_norm_plot requires epsilon_input in dataframe")
    d["epsilon_input"] = pd.to_numeric(d["epsilon_input"], errors="coerce")
    if "pgd_constrained_eps" not in d.columns:
        d["pgd_constrained_eps"] = [
            _compute_pgd_constrained_eps(eps_input=e, attack_norm=n) if pd.notna(e) else float("nan")
            for e, n in zip(d["epsilon_input"], d["attack_norm"])
        ]

    if x_col not in d.columns:
        raise ValueError(f"Missing x column for plotting: {x_col}")

    d = d.sort_values(x_col)

    x_vals = [float(v) for v in d[x_col].tolist()]
    y_vals = [float(v) for v in d["adv_top1"].tolist()]
    if "clean_top1" in d.columns and not d["clean_top1"].empty:
        clean_top1 = float(d["clean_top1"].iloc[0])
        if 0.0 not in x_vals:
            x_vals.append(0.0)
            y_vals.append(clean_top1)

    xy = sorted(zip(x_vals, y_vals), key=lambda t: t[0])
    x_sorted = [t[0] for t in xy]
    y_sorted = [t[1] for t in xy]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_sorted, y_sorted, marker="o", label=norm)
    ticks = sorted(set(x_sorted))
    _apply_x_scaling_and_ticks(ax, ticks, norm)
    ax.set_xlabel("PGD constrained eps" if x_col == "pgd_constrained_eps" else x_col)
    ax.set_ylabel("Top-1 Accuracy (%)")
    plt.title(f"{model_name}: {norm} PGD")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def main() -> None:
    args = parse_args()
    df = _load_dataframe(args)

    req_cols = {"attack_norm", "adv_top1", "clean_top1", "epsilon_input"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = _enrich_metadata(df, filter_madry_only=args.filter_madry_only)
    if args.x_col not in df.columns:
        raise ValueError(f"Missing required x-axis column after enrichment: {args.x_col}")
    df = df[df["train_norm"].isin(NORM_ORDER) & df["attack_norm"].isin(NORM_ORDER)]
    if df.empty:
        raise RuntimeError("No rows left after filtering to supported norms.")

    sub = "madry" if args.filter_madry_only else "all"
    prepared_csv = Path(args.out_dir) / sub / "prepared_plot_data.csv"
    prepared_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prepared_csv, index=False)

    saved: List[str] = []
    if args.make_matrix_plot:
        saved.append(save_norm_matrix_plot(df, args.out_dir, args.x_col, madry_only=args.filter_madry_only))

    if args.make_pair_plots:
        for attack_norm in NORM_ORDER:
            for train_norm in NORM_ORDER:
                p = save_pair_plot(df, args.out_dir, args.x_col, train_norm, attack_norm, madry_only=args.filter_madry_only)
                saved.append(p)

    print(f"Saved {len(saved)} plot(s).")
    print(f"Prepared CSV: {prepared_csv}")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
