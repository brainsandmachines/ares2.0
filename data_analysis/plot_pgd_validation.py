import argparse
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV = "data_analysis/pgd_validation_results.csv"
DEFAULT_OUT_DIR = "data_analysis/plots"
DEFAULT_MODELS_DIR = "/home/ashtomer/projects/ares/results/models"
NORM_ORDER = ["linf", "l2", "l1"]
HEATMAP_EVAL_ORDER = ["linf", "l2", "l1_apgd"]
EXCLUDED_NON_MADRY = ("gradnorm", "trades")
HEATMAP_EPS_ORDER = [1.0, 2.0, 4.0, 8.0, 16.0]
HEATMAP_MODEL_ORDER = ["baseline", 1.0, 2.0, 4.0, 8.0, 16.0]
PLOT_CATEGORIES = ["madry", "trades", "gradnorm"]
CATEGORY_TRAIN_FAMILIES = {
    "madry": ["linf", "l2", "l1"],
    "trades": ["linf", "l2"],
    "gradnorm": ["gradnorm"],
}
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
        help="Backward-compatible shorthand for plotting only madry models",
    )
    parser.add_argument(
        "--no-filter-madry-only",
        action="store_false",
        dest="filter_madry_only",
        help="Disable madry-only shorthand",
    )
    parser.add_argument(
        "--category",
        choices=PLOT_CATEGORIES,
        default=None,
        help="Plot exactly one training method.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=PLOT_CATEGORIES,
        default=None,
        help="Plot these training methods. Defaults to madry only unless --no-filter-madry-only is set.",
    )
    parser.add_argument("--make-matrix-plot", action="store_true", default=True, help="Save train-family/eval-norm line matrix")
    parser.add_argument(
        "--no-make-matrix-plot",
        action="store_false",
        dest="make_matrix_plot",
        help="Disable the train-family/eval-norm line matrix",
    )
    parser.add_argument("--make-pair-plots", action="store_true", default=True, help="Save per-(train_family,eval_norm) heatmaps")
    parser.add_argument(
        "--no-make-pair-plots",
        action="store_false",
        dest="make_pair_plots",
        help="Disable per-pair heatmaps",
    )
    parser.add_argument("--make-heatmap-grid", action="store_true", default=True, help="Save method-specific heatmap grid")
    parser.add_argument(
        "--no-make-heatmap-grid",
        action="store_false",
        dest="make_heatmap_grid",
        help="Disable method-specific heatmap grid",
    )
    parser.add_argument(
        "--include-inits",
        nargs="+",
        default=None,
        help="Restrict aggregate plots/prepared CSV to these init values, e.g. --include-inits 1 2",
    )
    return parser.parse_args()


def _resolve_categories(args: argparse.Namespace) -> List[str]:
    if args.category:
        return [args.category]
    if args.categories:
        return list(dict.fromkeys(args.categories))
    if args.filter_madry_only:
        return ["madry"]
    return PLOT_CATEGORIES[:]


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


def _parse_meta_from_model_dir(model_dir_name: str) -> Dict[str, object]:
    low = (model_dir_name or "").lower()
    is_baseline = "baseline" in low

    category = "unknown"
    if is_baseline:
        category = "baseline"
    elif "gradnorm" in low:
        category = "gradnorm"
    elif "trades" in low:
        category = "trades"
    elif re.search(r"(^|[_\-])(linf|l2|l1)([_\-]|$)", low):
        category = "madry"

    train_norm = "unknown"
    trades_match = re.search(r"(linf|l2|l1)trades(?:[_\-]|$)", low)
    if trades_match:
        train_norm = trades_match.group(1)
    else:
        norm_match = re.search(r"(^|[_\-])(linf|l2|l1)([_\-]|$)", low)
        if norm_match:
            train_norm = norm_match.group(2)

    train_family = train_norm
    if category == "gradnorm":
        train_family = "gradnorm"
    elif is_baseline:
        train_family = "baseline"

    init = "unknown"
    init_match = re.search(r"init[_\-]?(\d+)", low)
    if init_match:
        init = init_match.group(1)

    train_eps = None
    if not is_baseline:
        eps_match = None
        if category == "trades":
            eps_match = re.search(r"(?:linf|l2|l1)trades[_\-]?([0-9]*\.?[0-9]+)", low)
        elif category == "gradnorm":
            eps_match = re.search(r"gradnorm[_\-]?([0-9]*\.?[0-9]+)", low)
        else:
            eps_match = re.search(r"(?:linf|l2|l1)[_\-]?([0-9]*\.?[0-9]+)", low)
        if eps_match:
            try:
                train_eps = float(eps_match.group(1))
            except ValueError:
                train_eps = None

    return {
        "is_baseline": is_baseline,
        "category": category,
        "train_norm": train_norm,
        "train_family": train_family,
        "init": init,
        "train_eps": train_eps,
    }


def _extract_model_dir_from_row(row: pd.Series) -> str:
    model_dir_name = row.get("_model_dir_name", "")
    if isinstance(model_dir_name, str) and model_dir_name:
        return model_dir_name

    cp = row.get("checkpoint_path", "")
    if isinstance(cp, str) and cp:
        try:
            return Path(cp).parent.name
        except Exception:
            pass

    model_name = row.get("model_name", "")
    if isinstance(model_name, str) and model_name:
        return model_name

    return "unknown_model"


def _model_dir_from_csv_path(csv_path: Path) -> str:
    if csv_path.parent.name in {"pgd_eval", "pgd_eval_l1_apgd"}:
        return csv_path.parent.parent.name
    return csv_path.parent.name


def _discover_csvs(models_dir: str, filter_madry_only: bool) -> List[Path]:
    root = Path(models_dir)
    csvs = list(root.glob("*/pgd_eval/pgd_validation_results.csv"))
    csvs.extend(root.glob("*/pgd_eval/pgd_validation_results_l1_apgd.csv"))
    csvs.extend(root.glob("*/pgd_eval_l1_apgd/pgd_validation_results_l1_apgd.csv"))
    csvs.extend(root.glob("*/pgd_validation_results.csv"))
    csvs.extend(root.glob("*/pgd_validation_results_l1_apgd.csv"))
    csvs = sorted({p.resolve() for p in csvs})
    csvs = [Path(p) for p in csvs]
    if filter_madry_only:
        csvs = [p for p in csvs if _is_madry_dir(_model_dir_from_csv_path(p))]
    return csvs


def _source_kind_from_path(source_csv: str) -> str:
    p = Path(str(source_csv))
    parent = p.parent.name
    name = p.name
    if parent == "pgd_eval_l1_apgd" or name == "pgd_validation_results_l1_apgd.csv":
        return "l1_apgd"
    return "standard"


def _apply_per_model_attack_source_preference(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "_source_csv" not in df.columns or "_model_dir_name" not in df.columns:
        return df

    out_frames = []
    for _, dmodel in df.groupby("_model_dir_name", sort=False, dropna=False):
        dmodel = dmodel.copy()
        dmodel["_source_kind"] = dmodel["_source_csv"].map(_source_kind_from_path)
        dmodel["attack_norm"] = dmodel["attack_norm"].astype(str).str.lower()

        non_l1 = dmodel[dmodel["attack_norm"] != "l1"].copy()
        l1_rows = dmodel[dmodel["attack_norm"] == "l1"].copy()

        if l1_rows.empty:
            out_frames.append(non_l1)
            continue

        l1_apgd_rows = l1_rows[l1_rows["_source_kind"] == "l1_apgd"].copy()
        if not l1_apgd_rows.empty:
            out_frames.append(pd.concat([non_l1, l1_apgd_rows], ignore_index=True))
            continue

        standard_l1_rows = l1_rows[l1_rows["_source_kind"] == "standard"].copy()
        out_frames.append(pd.concat([non_l1, standard_l1_rows], ignore_index=True))

    if not out_frames:
        return df.iloc[0:0].copy()

    out = pd.concat(out_frames, ignore_index=True)
    if "_source_kind" in out.columns:
        out = out.drop(columns=["_source_kind"])
    return out


def _load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.auto_collect_pgd_csvs:
        requested_categories = bool(args.category or args.categories)
        csv_paths = _discover_csvs(args.models_dir, args.filter_madry_only and not requested_categories)
        if not csv_paths:
            raise RuntimeError(f"No PGD CSV files found under {args.models_dir}")

        frames = []
        for p in csv_paths:
            d = pd.read_csv(p)
            if d.empty:
                continue
            d["_source_csv"] = str(p)
            d["_model_dir_name"] = _model_dir_from_csv_path(p)
            frames.append(d)

        if not frames:
            raise RuntimeError("All discovered PGD CSV files were empty.")

        df = pd.concat(frames, ignore_index=True)
        return _apply_per_model_attack_source_preference(df)

    return pd.read_csv(args.csv)


def _normalize_known_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    model_dirs = [_extract_model_dir_from_row(row) for _, row in df.iterrows()]
    df["model_dir"] = model_dirs

    parsed = [_parse_meta_from_model_dir(m) for m in model_dirs]
    df["is_baseline"] = [bool(p["is_baseline"]) for p in parsed]
    df["_parsed_category"] = [str(p["category"]) for p in parsed]
    df["_parsed_train_norm"] = [str(p["train_norm"]) for p in parsed]
    df["_parsed_train_family"] = [str(p["train_family"]) for p in parsed]
    df["_parsed_init"] = [str(p["init"]) for p in parsed]
    df["_parsed_train_eps"] = [p["train_eps"] for p in parsed]

    if "train_norm" not in df.columns:
        df["train_norm"] = "unknown"
    if "init" not in df.columns:
        df["init"] = "unknown"
    if "category" not in df.columns:
        df["category"] = "unknown"
    if "l1_step_mode" not in df.columns:
        df["l1_step_mode"] = ""

    df["train_norm"] = df["train_norm"].astype("object")
    df["init"] = df["init"].astype("object")
    df["category"] = df["category"].astype("object")
    df["l1_step_mode"] = df["l1_step_mode"].fillna("").astype(str).str.lower()

    known_train_norm = _normalize_known_str(df["train_norm"]).isin(NORM_ORDER)
    df.loc[~known_train_norm, "train_norm"] = df.loc[~known_train_norm, "_parsed_train_norm"]

    parsed_init = _normalize_known_str(df["_parsed_init"])
    df.loc[parsed_init.ne("") & ~parsed_init.isin(["unknown", "nan", "none"]), "init"] = df.loc[
        parsed_init.ne("") & ~parsed_init.isin(["unknown", "nan", "none"]),
        "_parsed_init",
    ]

    category_known = _normalize_known_str(df["category"]).isin(PLOT_CATEGORIES)
    df.loc[~category_known, "category"] = df.loc[~category_known, "_parsed_category"]
    df["category"] = _normalize_known_str(df["category"])

    df["train_family"] = df["_parsed_train_family"].astype(str)
    df.loc[df["is_baseline"], "train_family"] = "baseline"

    if "train_eps" in df.columns:
        existing_train_eps = pd.to_numeric(df["train_eps"], errors="coerce")
    else:
        existing_train_eps = pd.Series(np.nan, index=df.index)
    parsed_train_eps = pd.to_numeric(df["_parsed_train_eps"], errors="coerce")
    df["train_eps"] = existing_train_eps.where(existing_train_eps.notna(), parsed_train_eps)

    if "attack_norm" not in df.columns:
        raise ValueError("Missing required column: attack_norm")
    df["attack_norm"] = _normalize_known_str(df["attack_norm"])

    if "epsilon_input" not in df.columns:
        raise ValueError("Missing required column: epsilon_input")
    df["epsilon_input"] = pd.to_numeric(df["epsilon_input"], errors="coerce")
    df["pgd_constrained_eps"] = [
        _compute_pgd_constrained_eps(eps_input=e, attack_norm=n) if pd.notna(e) else float("nan")
        for e, n in zip(df["epsilon_input"], df["attack_norm"])
    ]

    df["model_id"] = df["model_dir"].astype(str)
    df["eval_variant"] = np.where(
        (df["attack_norm"] == "l1") & (df["l1_step_mode"] == "l1_apgd"),
        "l1_apgd",
        df["attack_norm"],
    )

    labels = []
    for is_baseline, eps, init in zip(df["is_baseline"], df["train_eps"], df["init"]):
        if is_baseline:
            labels.append("baseline")
            continue
        init_str = str(init)
        if init_str.lower() in ("unknown", "nan", "none", ""):
            labels.append(f"eps={_trim_float(eps)}")
        else:
            labels.append(f"eps={_trim_float(eps)} init {init_str}")
    df["model_label"] = labels
    return df


def _expand_baseline_rows(df: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    baseline_mask = df["is_baseline"].fillna(False)
    if not baseline_mask.any():
        return df

    baseline = df[baseline_mask].copy()
    non_baseline = df[~baseline_mask].copy()

    expanded = []
    for category in categories:
        for train_family in CATEGORY_TRAIN_FAMILIES.get(category, []):
            dcopy = baseline.copy()
            dcopy["plot_category"] = category
            dcopy["train_family"] = train_family
            expanded.append(dcopy)

    out = non_baseline.copy()
    out["plot_category"] = out["category"]
    if expanded:
        out = pd.concat([out, *expanded], ignore_index=True)
    return out


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


def _heatmap_model_level(row: pd.Series):
    if bool(row.get("is_baseline", False)):
        return "baseline"

    eps = pd.to_numeric(row.get("train_eps"), errors="coerce")
    if pd.isna(eps):
        return np.nan

    eps = float(eps)
    if eps in HEATMAP_EPS_ORDER:
        return eps
    return np.nan


def _build_heatmap_pivot(dpair: pd.DataFrame) -> pd.DataFrame:
    pivot = dpair.pivot_table(
        index="epsilon_input",
        columns="heatmap_model_level",
        values="adv_top1",
        aggfunc="mean",
    )
    return pivot.reindex(index=HEATMAP_EPS_ORDER, columns=HEATMAP_MODEL_ORDER)


def _build_heatmap_with_clean_row(dpair: pd.DataFrame):
    dpair = dpair.copy()
    dpair["epsilon_input"] = pd.to_numeric(dpair["epsilon_input"], errors="coerce")
    dpair["heatmap_model_level"] = dpair.apply(_heatmap_model_level, axis=1)
    dpair = dpair[
        dpair["heatmap_model_level"].notna()
        & dpair["epsilon_input"].isin(HEATMAP_EPS_ORDER)
    ]

    pivot = _build_heatmap_pivot(dpair)
    if pivot.empty or pivot.isna().all().all():
        return None, None, None

    clean_by_train = (
        dpair.groupby("heatmap_model_level", dropna=True)["clean_top1"]
        .mean()
        .reindex(HEATMAP_MODEL_ORDER)
    )

    data = np.vstack([clean_by_train.to_numpy(dtype=float), pivot.to_numpy(dtype=float)])
    xlabels = ["baseline", "1", "2", "4", "8", "16"]
    ylabels = ["clean", "1", "2", "4", "8", "16"]
    return data, xlabels, ylabels


def _select_heatmap_pair_data(df: pd.DataFrame, train_family: str, eval_variant: str) -> pd.DataFrame:
    base = df[df["train_family"] == train_family].copy()
    if eval_variant != "l1_apgd":
        return base[base["eval_variant"] == eval_variant].copy()

    apgd = base[base["eval_variant"] == "l1_apgd"].copy()
    fallback = base[base["attack_norm"] == "l1"].copy()
    if fallback.empty:
        return apgd

    apgd_model_ids = set(apgd["model_id"].astype(str).tolist())
    fallback = fallback[~fallback["model_id"].astype(str).isin(apgd_model_ids)].copy()
    if apgd.empty:
        return fallback
    if fallback.empty:
        return apgd
    return pd.concat([apgd, fallback], ignore_index=True)


def _render_heatmap(ax, data: np.ndarray, xlabels: List[str], ylabels: List[str]):
    im = ax.imshow(data, origin="lower", aspect="equal", cmap="turbo", vmin=0, vmax=100)

    xticks = list(range(len(xlabels)))
    yticks = list(range(len(ylabels)))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = float(data[i, j])
            if np.isnan(val):
                continue
            txt_color = "white" if val < 20.0 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=7)
    return im


def _grid_shape_tag(n_rows: int, n_cols: int) -> str:
    return f"{n_rows}x{n_cols}"


def _build_output_path(out_dir: str, category: str, group: str, filename: str, subset_tag: str = "") -> Path:
    root = Path(out_dir) / category / group
    if subset_tag:
        root = root / "by_init" / subset_tag
    return root / filename


def save_heatmap_matrix_plot(df: pd.DataFrame, out_dir: str, category: str, subset_tag: str = "") -> str:
    train_families = CATEGORY_TRAIN_FAMILIES[category]
    n_rows = len(train_families)
    n_cols = len(HEATMAP_EVAL_ORDER)
    out_path = _build_output_path(
        out_dir,
        category,
        "norm_matrix",
        f"pgd_train_eval_eps_heatmap_{_grid_shape_tag(n_rows, n_cols)}.png",
        subset_tag=subset_tag,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 4.1 * n_rows),
        sharex=False,
        sharey=False,
        squeeze=False,
        constrained_layout=True,
    )
    im = None

    for row, train_family in enumerate(train_families):
        for col, eval_variant in enumerate(HEATMAP_EVAL_ORDER):
            ax = axes[row, col]
            dpair = _select_heatmap_pair_data(df, train_family, eval_variant)
            ax.set_title(f"trained on: {train_family} eval on: {eval_variant}")

            if dpair.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                data, xlabels, ylabels = _build_heatmap_with_clean_row(dpair)
                if data is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                else:
                    im = _render_heatmap(ax, data, xlabels, ylabels)

            if row == n_rows - 1:
                ax.set_xlabel("Trained constrained eps")
            if col == 0:
                ax.set_ylabel("Evaluated constrained eps")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.92, pad=0.02)
        cbar.set_label("Adv Top-1 Accuracy (%)")

    fig.suptitle(f"PGD Accuracy Heatmap Matrix ({category})", fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _plot_pair_subplot(ax, dpair: pd.DataFrame, train_family: str, attack_norm: str, x_col: str, xticks: List[float]) -> None:
    ax.set_title(f"train={train_family}, eval={attack_norm}")

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
        dmodel = dpair[dpair["model_id"] == model_id].copy()
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


def save_norm_matrix_plot(df: pd.DataFrame, out_dir: str, x_col: str, category: str, subset_tag: str = "") -> str:
    train_families = CATEGORY_TRAIN_FAMILIES[category]
    n_rows = len(train_families)
    n_cols = len(HEATMAP_EVAL_ORDER)
    out_path = _build_output_path(
        out_dir,
        category,
        "norm_matrix",
        f"pgd_train_vs_eval_{_grid_shape_tag(n_rows, n_cols)}.png",
        subset_tag=subset_tag,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    norm_ticks = _build_norm_ticks(df, x_col)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.2 * n_rows), sharey=True, squeeze=False)
    for row, train_family in enumerate(train_families):
        for col, attack_norm in enumerate(HEATMAP_EVAL_ORDER):
            ax = axes[row, col]
            dpair = _select_heatmap_pair_data(df, train_family, attack_norm)
            _plot_pair_subplot(ax, dpair, train_family, attack_norm, x_col, norm_ticks.get("l1" if attack_norm == "l1_apgd" else attack_norm, []))

            if row == n_rows - 1:
                ax.set_xlabel("PGD constrained eps" if x_col == "pgd_constrained_eps" else x_col)
            if col == 0:
                ax.set_ylabel("Top-1 Accuracy (%)")

    fig.suptitle(f"PGD Robustness Matrix ({category})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def save_pair_plot(
    df: pd.DataFrame,
    out_dir: str,
    category: str,
    train_family: str,
    attack_norm: str,
    subset_tag: str = "",
) -> str:
    out_path = _build_output_path(
        out_dir,
        category,
        "pairs",
        f"train_{train_family}__eval_{attack_norm}.png",
        subset_tag=subset_tag,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpair = _select_heatmap_pair_data(df, train_family, attack_norm).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"trained on: {train_family} eval on: {attack_norm}")

    if dpair.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        data, xlabels, ylabels = _build_heatmap_with_clean_row(dpair)
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            _render_heatmap(ax, data, xlabels, ylabels)

    ax.set_xlabel("Trained constrained eps")
    ax.set_ylabel("Evaluated constrained eps")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


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


def _filter_supported_rows(df: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    df = df[df["attack_norm"].isin(NORM_ORDER)].copy()
    df = df[df["epsilon_input"].isin(HEATMAP_EPS_ORDER) | df["epsilon_input"].eq(0.0)].copy()
    df = df[df["plot_category"].isin(categories)].copy()
    if df.empty:
        return df

    allowed_train_families = {family for category in categories for family in CATEGORY_TRAIN_FAMILIES[category]}
    df = df[df["train_family"].isin(allowed_train_families)].copy()
    return df


def _sorted_init_values(df: pd.DataFrame) -> List[str]:
    return sorted(
        {
            str(v)
            for v in df["init"].dropna().astype(str).tolist()
            if str(v).strip().lower() not in ("", "unknown", "nan", "none")
        },
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )


def _save_category_outputs(df: pd.DataFrame, args: argparse.Namespace, category: str) -> List[str]:
    saved: List[str] = []
    prepared_csv = Path(args.out_dir) / category / "prepared_plot_data.csv"
    prepared_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prepared_csv, index=False)
    saved.append(str(prepared_csv))

    train_families = CATEGORY_TRAIN_FAMILIES[category]
    if args.make_matrix_plot:
        saved.append(save_norm_matrix_plot(df, args.out_dir, args.x_col, category))

    if args.make_pair_plots:
        for train_family in train_families:
            for attack_norm in HEATMAP_EVAL_ORDER:
                saved.append(save_pair_plot(df, args.out_dir, category, train_family, attack_norm))

    if args.make_heatmap_grid:
        saved.append(save_heatmap_matrix_plot(df, args.out_dir, category))

    for init in _sorted_init_values(df):
        d_init = df[df["init"].astype(str) == init].copy()
        if d_init.empty:
            continue
        tag = f"init{init}"
        if args.make_matrix_plot:
            saved.append(save_norm_matrix_plot(d_init, args.out_dir, args.x_col, category, subset_tag=tag))
        if args.make_pair_plots:
            for train_family in train_families:
                for attack_norm in HEATMAP_EVAL_ORDER:
                    saved.append(save_pair_plot(d_init, args.out_dir, category, train_family, attack_norm, subset_tag=tag))
        if args.make_heatmap_grid:
            saved.append(save_heatmap_matrix_plot(d_init, args.out_dir, category, subset_tag=tag))

    return saved


def main() -> None:
    args = parse_args()
    categories = _resolve_categories(args)
    df = _load_dataframe(args)

    req_cols = {"attack_norm", "adv_top1", "clean_top1", "epsilon_input"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = _enrich_metadata(df)
    df = _expand_baseline_rows(df, categories)
    df["epsilon_input"] = pd.to_numeric(df["epsilon_input"], errors="coerce")
    if args.x_col not in df.columns:
        raise ValueError(f"Missing required x-axis column after enrichment: {args.x_col}")
    if args.include_inits:
        allowed_inits = {str(v).strip() for v in args.include_inits if str(v).strip()}
        df = df[df["init"].astype(str).isin(allowed_inits)]

    df = _filter_supported_rows(df, categories)
    if df.empty:
        raise RuntimeError("No rows left after filtering to supported methods and norms.")

    saved: List[str] = []
    for category in categories:
        dcat = df[df["plot_category"] == category].copy()
        if dcat.empty:
            continue
        saved.extend(_save_category_outputs(dcat, args, category))

    if not saved:
        raise RuntimeError("No outputs were produced for the requested categories.")

    print(f"Saved {len(saved)} output(s).")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
