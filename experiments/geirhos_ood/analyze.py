#!/usr/bin/env python3
"""Connections between training protocol, threat norm, epsilon, shape bias and the 16 OOD tasks.

Joins, one row per checkpoint (one per run; the set is ``models.load_models()``, 201 today):
- ``results/task_metrics.csv``: 16-class OOD accuracy and error consistency per task
- the five contstim zoo JSONs: arch, protocol, norm, eps, full-val clean accuracy (bilinear resize, the same
  preprocessing as this evaluation), AA robust accuracy at the trained threat
- ``data_analysis/shape_bias_results_botero.csv``: cue-conflict shape bias (NaN where not
  scored, e.g. the vit_b_cvst / swin_b TRADES checkpoints)

Outputs ``analysis/tables/*.csv`` and ``analysis/plots/*.{pdf,png}`` (findings: ``analysis/FINDINGS.md``):
- ``eps_curves_<group>``: each task's accuracy against eps, Madry (solid) vs TRADES (dashed), one
  colour per norm, for convnext_base, convnext_base DVD-B, vit_b_cvst and swin_b
- ``madry_minus_trades``: Madry minus TRADES at matched (arch, norm, eps), averaged over eps
- ``eps_change``: fitted change from eps 1 to eps 8 per (group, protocol, norm) and task
- ``clean_acc_control``: is the protocol effect only a clean-accuracy effect? OLS per task of
  accuracy ~ clean accuracy + TRADES indicator (+ norm and arch dummies), and scatters
- ``robustness_matched``: matched-eps Madry − TRADES gaps against their AA robust-accuracy gap
- ``shape_bias_correlations``: Spearman of shape bias with each task (all, within Madry,
  within TRADES, partial on clean accuracy), plus scatters
- ``task_correlations``: Spearman matrix between the tasks and the model-level measures

eps is the level in the run name (1, 2, 4, 6, 8). Levels 6 and 8 are continuation runs that
start from the eps-4 model, so the top of each curve also carries the continuation protocol.

    python -m experiments.geirhos_ood.analyze
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster import hierarchy

from experiments.geirhos_ood.models import load_models, zoo_entries
from experiments.geirhos_ood.tasks import TASKS

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
ANALYSIS_DIR = EXP_DIR / "analysis"
OUT_TABLES = ANALYSIS_DIR / "tables"
OUT_PLOTS = ANALYSIS_DIR / "plots"
SHAPE_BIAS_CSV = EXP_DIR.parents[1] / "data_analysis" / "shape_bias_results_botero.csv"

TASK_IDS = list(TASKS)
NORMS = ("l1", "l2", "linf")
NORM_COLOR = {"l1": "#4C72B0", "l2": "#DD8452", "linf": "#55A868"}
PROTOCOL_STYLE = {"madry": "-", "trades": "--"}
PROTOCOL_COLOR = {"madry": "#C44E52", "trades": "#4C72B0", "clean": "#333333", "other": "#999999"}
GROUPS = {  # name -> (arch, variant)
    "convnext_base": ("convnext_base", "standard"),
    "convnext_base_dvdb": ("convnext_base", "dvd_b"),
    "vit_b_cvst": ("vit_b_cvst", "standard"),
    "swin_b": ("swin_b", "standard"),
}
EXTRA = {"clean_acc": "clean acc (full val)", "aa_robust_acc": "AA robust acc", "shape_bias": "shape bias",
         "mean_ood": "mean OOD acc (16)"}


def _variant(meta: dict) -> str:
    return {"madry_trades": "standard", "dvd_b": "dvd_b", "v1": "v1", "gradnorm": "gradnorm"}.get(
        meta["protocol"], "baseline")


def model_table() -> pd.DataFrame:
    """One row per checkpoint: metadata, clean/AA accuracy, shape bias, 16 accuracies and ECs."""
    meta = {m["checkpoint"]: m for _source, m in zoo_entries()}
    rows = []
    for e in load_models():
        m = meta[str(e.checkpoint)]
        norm = m["norm"]
        rows.append({
            "model_id": e.model_id, "family": e.family, "arch": e.arch, "variant": _variant(m),
            "protocol": ("clean" if norm is None else "trades" if "trades" in norm else "madry")
            if m["protocol"] in ("madry_trades", "dvd_b", "clean_baseline") else "other",
            "norm": norm.replace("trades", "") if norm else None, "eps": m["eps"], "init": e.init,
            "kind": e.kind, "generator": e.generator, "clean_acc": m["clean_acc_bilinear"] / 100,
            "aa_robust_acc": m["aa_robust_acc"] / 100 if m["aa_robust_acc"] is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    sb = pd.read_csv(SHAPE_BIAS_CSV)
    sb = sb.assign(model_id=sb.model_name + "__" + sb.checkpoint).set_index("model_id")
    df["shape_bias"] = df.model_id.map(sb.score)
    tm = pd.read_csv(RESULTS_DIR / "task_metrics.csv")
    tm = tm[tm.status == "done"]
    acc = tm.pivot(index="model_id", columns="task_id", values="ood_accuracy")[TASK_IDS]
    ec = tm.pivot(index="model_id", columns="task_id", values="error_consistency")[TASK_IDS]
    df = df.join(acc, on="model_id").join(ec.add_prefix("ec_"), on="model_id")
    df["mean_ood"] = df[TASK_IDS].mean(axis=1)
    df["mean_ec"] = df[[f"ec_{t}" for t in TASK_IDS]].mean(axis=1)
    if df[TASK_IDS].isna().any().any():
        raise ValueError("task_metrics.csv is missing cells; run the evaluation first")
    return df


def group_rows(df: pd.DataFrame, group: str) -> pd.DataFrame:
    arch, variant = GROUPS[group]
    return df[(df.arch == arch) & (df.variant == variant) & df.protocol.isin(["madry", "trades"])]


def baseline_rows(df: pd.DataFrame, group: str) -> pd.DataFrame:
    arch, variant = GROUPS[group]
    base = df[(df.arch == arch) & (df.protocol == "clean") & (df.variant == "baseline")]
    dvd = base.family.str.contains("dvd") | base.model_id.str.contains("dvd_b")
    return base[dvd] if variant == "dvd_b" else base[~dvd]


def human_accuracy() -> dict[str, float]:
    h = pd.read_csv(RESULTS_DIR / "human_metrics.csv")
    return dict(zip(h.task_id, h.ood_accuracy))


# ---------------------------------------------------------------- eps curves

def plot_eps_curves(df: pd.DataFrame, group: str, human: dict[str, float]) -> plt.Figure:
    sub, base = group_rows(df, group), baseline_rows(df, group)
    panels = TASK_IDS + list(EXTRA)
    fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharex=True)
    for ax, col in zip(axes.flat, panels):
        for (proto, norm), g in sub.groupby(["protocol", "norm"]):
            mean = g.groupby("eps")[col].mean()
            ax.plot(mean.index, mean.values, PROTOCOL_STYLE[proto], color=NORM_COLOR[norm], marker="o", ms=3, lw=1.6)
            ax.scatter(g.eps, g[col], color=NORM_COLOR[norm], s=6, alpha=0.35)
        if base[col].notna().any():
            ax.axhline(base[col].mean(), color="grey", ls=":", lw=1.4)
        if col in human:
            ax.axhline(human[col], color="red", ls="--", lw=1)
        ax.set_title(TASKS[col].label if col in TASKS else EXTRA[col], fontsize=10)
        ax.set_xticks([1, 2, 4, 6, 8])
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("eps level")
    handles = [Line2D([], [], color=NORM_COLOR[n], lw=2, label=n) for n in NORMS] + [
        Line2D([], [], color="k", ls=PROTOCOL_STYLE[p], label=p) for p in PROTOCOL_STYLE] + [
        Line2D([], [], color="grey", ls=":", label="clean baseline"), Line2D([], [], color="red", ls="--", label="humans")]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"{group}: accuracy vs eps (line = mean over checkpoints at that eps, dots = checkpoints)",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Madry - TRADES

def madry_minus_trades(df: pd.DataFrame) -> pd.DataFrame:
    cols = TASK_IDS + ["mean_ood", "clean_acc", "aa_robust_acc", "shape_bias", "mean_ec"]
    rows = []
    for group in GROUPS:
        sub = group_rows(df, group)
        for norm in NORMS:
            cell = sub[sub.norm == norm].groupby(["protocol", "eps"])[cols].mean()
            if not {"madry", "trades"} <= set(cell.index.get_level_values(0)):
                continue
            eps = sorted(set(cell.loc["madry"].index) & set(cell.loc["trades"].index))
            diff = cell.loc["madry"].loc[eps] - cell.loc["trades"].loc[eps]
            rows.append({"group": group, "norm": norm, "n_eps": len(eps), **diff.mean().to_dict()})
    return pd.DataFrame(rows)


def plot_heatmap(table: pd.DataFrame, cols: list[str], row_labels: list[str], title: str, cbar: str,
                 vlim: float | None = None) -> plt.Figure:
    data = table[cols].to_numpy(dtype=float)
    vlim = vlim or np.nanmax(np.abs(data))
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 3, 0.42 * len(row_labels) + 2.5))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    for (r, c), v in np.ndenumerate(data):
        if np.isfinite(v):
            ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 * vlim else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([TASKS[c].label if c in TASKS else EXTRA.get(c, c) for c in cols], rotation=45, ha="left")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax, shrink=0.6, label=cbar)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- eps change

def eps_change(df: pd.DataFrame) -> pd.DataFrame:
    """Linear fit of value on eps level within each (group, protocol, norm); report fitted eps1 -> eps8 change."""
    cols = TASK_IDS + ["mean_ood", "clean_acc", "shape_bias", "mean_ec"]
    rows = []
    for group in GROUPS:
        for (proto, norm), g in group_rows(df, group).groupby(["protocol", "norm"]):
            row = {"group": group, "protocol": proto, "norm": norm, "n": len(g)}
            for c in cols:
                ok = g[c].notna()
                row[c] = np.polyfit(g.eps[ok], g[c][ok], 1)[0] * 7 if ok.sum() >= 3 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- clean-accuracy control

def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


def clean_acc_control(df: pd.DataFrame) -> pd.DataFrame:
    """Per task, over every Madry/TRADES checkpoint (standard + DVD-B, 3 archs):
    acc ~ 1 + trades + dvd dummy + norm dummies + arch dummies [+ clean_acc [+ eps dummies + AA robust acc per norm]].
    Reports the TRADES coefficient for the three specifications. AA is measured at each run's own
    threat, so it is only comparable at the same eps: hence the eps dummies and one slope per norm. At matched eps Madry is the more robust protocol, so the
    last specification is the one that compares the two at equal clean accuracy *and* robustness."""
    adv = df[df.protocol.isin(["madry", "trades"]) & df.variant.isin(["standard", "dvd_b"])]
    dummies = pd.get_dummies(adv[["norm", "arch"]], drop_first=True).astype(float)
    base = np.column_stack([np.ones(len(adv)), (adv.protocol == "trades").astype(float),
                            (adv.variant == "dvd_b").astype(float), dummies.to_numpy()])
    aa_by_norm = np.column_stack([adv.aa_robust_acc.fillna(0) * (adv.norm == n) for n in NORMS])
    eps_dummies = pd.get_dummies(adv.eps.astype(int), drop_first=True).astype(float).to_numpy()
    rows = []
    for c in TASK_IDS + ["mean_ood", "shape_bias", "mean_ec"]:
        ok = adv[c].notna().to_numpy()
        y = adv[c].to_numpy(dtype=float)
        b0, se0 = ols(y[ok], base[ok])
        X = np.column_stack([base, adv.clean_acc.to_numpy()])
        b1, se1 = ols(y[ok], X[ok])
        ok2 = ok & adv.aa_robust_acc.notna().to_numpy()
        b2, se2 = ols(y[ok2], np.column_stack([X, eps_dummies, aa_by_norm])[ok2])
        rows.append({"measure": c, "n": int(ok.sum()), "trades_effect_raw": b0[1], "trades_se_raw": se0[1],
                     "trades_effect_given_clean": b1[1], "trades_se_given_clean": se1[1],
                     "clean_acc_slope": b1[-1], "clean_acc_slope_se": se1[-1], "n_with_aa": int(ok2.sum()),
                     "trades_effect_given_clean_aa": b2[1], "trades_se_given_clean_aa": se2[1]})
    return pd.DataFrame(rows)


def matched_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Madry minus TRADES per matched (group, norm, eps), each side averaged over its checkpoints."""
    cols = TASK_IDS + ["mean_ood", "clean_acc", "aa_robust_acc", "shape_bias", "mean_ec"]
    rows = []
    for group in GROUPS:
        cell = group_rows(df, group).groupby(["norm", "eps", "protocol"])[cols].mean()
        for (norm, eps), g in cell.groupby(level=[0, 1]):
            g = g.droplevel([0, 1])
            if {"madry", "trades"} <= set(g.index):
                rows.append({"group": group, "norm": norm, "eps": eps, **(g.loc["madry"] - g.loc["trades"]).to_dict()})
    return pd.DataFrame(rows)


def plot_robustness_matched(pairs: pd.DataFrame, cols: list[str]) -> plt.Figure:
    """Each point is one matched-eps Madry/TRADES pair: x = AA robust gap at that eps (same threat for
    both), y = task gap. The fitted line's value at x = 0 is the gap at equal robustness."""
    markers = {"convnext_base": "o", "convnext_base_dvdb": "D", "vit_b_cvst": "^", "swin_b": "s"}
    fig, axes = plt.subplots(2, -(-len(cols) // 2), figsize=(3.6 * -(-len(cols) // 2), 7.4), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        for (group, norm), g in pairs.groupby(["group", "norm"]):
            ax.scatter(g.aa_robust_acc, g[c], s=22, marker=markers[group], color=NORM_COLOR[norm], alpha=0.8)
        ok = pairs[["aa_robust_acc", c]].notna().all(axis=1)
        x, y = pairs.aa_robust_acc[ok], pairs[c][ok]
        fit = stats.linregress(x, y)
        xs = np.linspace(min(0, x.min()), x.max(), 20)
        ax.plot(xs, fit.intercept + fit.slope * xs, "k-", lw=1)
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(f"{TASKS[c].label if c in TASKS else EXTRA.get(c, c)}\n"
                     f"at equal robustness: {fit.intercept:+.3f} ± {fit.intercept_stderr:.3f}", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Madry − TRADES, AA robust acc", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Madry − TRADES, accuracy")
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    handles = [Line2D([], [], ls="", marker="o", color=NORM_COLOR[n], label=n) for n in NORMS] + [
        Line2D([], [], ls="", marker=m, color="k", label=g) for g, m in markers.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Madry − TRADES at matched (arch, norm, eps): is the gap explained by Madry being more robust?",
                 y=1.04, fontsize=13)
    fig.tight_layout()
    return fig


def plot_clean_scatter(df: pd.DataFrame, cols: list[str], x: str, xlabel: str, title: str) -> plt.Figure:
    n = len(cols)
    ncol = 4
    fig, axes = plt.subplots(-(-n // ncol), ncol, figsize=(4.2 * ncol, 3.6 * -(-n // ncol)), squeeze=False)
    markers = {"l1": "o", "l2": "s", "linf": "^", None: "*"}
    for ax, c in zip(axes.flat, cols):
        for (proto, norm), g in df.groupby(["protocol", df.norm.fillna("none")], dropna=False):
            ax.scatter(g[x], g[c], s=14 if proto != "clean" else 60, alpha=0.75, color=PROTOCOL_COLOR[proto],
                       marker=markers.get(None if norm == "none" else norm, "o"), edgecolor="none")
        ok = df[[x, c]].notna().all(axis=1)
        rho = stats.spearmanr(df.loc[ok, x], df.loc[ok, c])[0]
        ax.set_title(f"{TASKS[c].label if c in TASKS else EXTRA.get(c, c)}  (rho={rho:+.2f})", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    handles = [Line2D([], [], ls="", marker="o", color=PROTOCOL_COLOR[p], label=p) for p in PROTOCOL_COLOR] + [
        Line2D([], [], ls="", marker=markers[k], color="k", label=k) for k in NORMS]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- shape-bias correlations

def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, float]:
    rx, ry, rz = (s.rank().to_numpy() for s in (x, y, z))
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), n - 3)


def shape_bias_correlations(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.shape_bias.notna()]
    subsets = {"all": d, "madry": d[d.protocol == "madry"], "trades": d[d.protocol == "trades"]}
    rows = []
    for c in TASK_IDS + [f"ec_{t}" for t in TASK_IDS] + ["mean_ood", "mean_ec", "clean_acc", "aa_robust_acc"]:
        row = {"measure": c}
        for name, s in subsets.items():
            ok = s[c].notna()
            r, p = stats.spearmanr(s.shape_bias[ok], s[c][ok])
            row.update({f"rho_{name}": r, f"p_{name}": p, f"n_{name}": int(ok.sum())})
        if c != "clean_acc":
            ok = d[c].notna()
            row["partial_rho_given_clean"], row["partial_p_given_clean"] = partial_spearman(
                d.shape_bias[ok], d[c][ok], d.clean_acc[ok])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_shape_bias_bars(corr: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharey=True)
    series = [("rho_all", "all (n=169)", "#333333"), ("partial_rho_given_clean", "all, partial on clean acc", "#8172B2"),
              ("rho_madry", "within Madry", PROTOCOL_COLOR["madry"]), ("rho_trades", "within TRADES", PROTOCOL_COLOR["trades"])]
    for ax, prefix, title in ((axes[0], "", "shape bias vs OOD accuracy"),
                              (axes[1], "ec_", "shape bias vs error consistency with humans")):
        names = [f"{prefix}{t}" for t in TASK_IDS] + (["mean_ood", "clean_acc", "aa_robust_acc"] if not prefix else ["mean_ec"])
        sub = corr.set_index("measure").loc[names]
        x = np.arange(len(names))
        for i, (col, label, color) in enumerate(series):
            ax.bar(x + (i - 1.5) * 0.2, sub[col], 0.2, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([TASKS[n.removeprefix("ec_")].label if n.removeprefix("ec_") in TASKS
                            else EXTRA.get(n, n) for n in names], rotation=35, ha="right", fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Spearman rho")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=4, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- task x task

def task_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = TASK_IDS + ["shape_bias", "clean_acc", "aa_robust_acc", "mean_ec"]
    return df[cols].corr(method="spearman")


def plot_corr_matrix(corr: pd.DataFrame) -> plt.Figure:
    order = hierarchy.leaves_list(hierarchy.linkage(1 - corr.fillna(0).to_numpy()[np.triu_indices(len(corr), 1)],
                                                    "average"))
    c = corr.iloc[order, order]
    labels = [TASKS[n].label if n in TASKS else EXTRA.get(n, "mean error consistency") for n in c.index]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    for (r, k), v in np.ndenumerate(c.to_numpy()):
        ax.text(k, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho across checkpoints")
    fig.suptitle("Which measures move together across the checkpoints (clustered order)", fontsize=13)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT_PLOTS / f"{name}.{ext}", dpi=110 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    df = model_table()
    df.to_csv(OUT_TABLES / "model_table.csv", index=False)
    human = human_accuracy()

    for group in GROUPS:
        save(plot_eps_curves(df, group, human), f"eps_curves_{group}")

    heat_cols = TASK_IDS + ["mean_ood", "clean_acc", "aa_robust_acc", "shape_bias"]
    mmt = madry_minus_trades(df)
    mmt.to_csv(OUT_TABLES / "madry_minus_trades.csv", index=False)
    save(plot_heatmap(mmt, heat_cols, [f"{g} {n}" for g, n in zip(mmt.group, mmt.norm)],
                      "Madry minus TRADES at matched (arch, norm, eps), mean over eps", "accuracy difference"),
         "madry_minus_trades")

    ch = eps_change(df)
    ch.to_csv(OUT_TABLES / "eps_change.csv", index=False)
    save(plot_heatmap(ch, TASK_IDS + ["mean_ood", "clean_acc", "shape_bias"],
                      [f"{g} {p} {n}" for g, p, n in zip(ch.group, ch.protocol, ch.norm)],
                      "Fitted change from eps 1 to eps 8 (linear fit within each protocol x norm)", "change"),
         "eps_change")

    ctrl = clean_acc_control(df)
    ctrl.to_csv(OUT_TABLES / "clean_acc_control.csv", index=False)
    save(plot_clean_scatter(df, TASK_IDS, "clean_acc", "clean accuracy (full val, 1000-class)",
                            "Task accuracy vs clean accuracy, by protocol"), "clean_acc_scatter")
    pairs = matched_pairs(df)
    pairs.to_csv(OUT_TABLES / "madry_minus_trades_pairs.csv", index=False)
    save(plot_robustness_matched(pairs, ["contrast", "uniform-noise", "high-pass", "power-equalisation", "phase-scrambling",
                                      "stylized", "edge", "silhouette", "shape_bias", "mean_ood"]), "robustness_matched")

    sbc = shape_bias_correlations(df)
    sbc.to_csv(OUT_TABLES / "shape_bias_correlations.csv", index=False)
    save(plot_shape_bias_bars(sbc), "shape_bias_correlations")
    save(plot_clean_scatter(df[df.shape_bias.notna()], TASK_IDS + ["clean_acc", "mean_ood"], "shape_bias",
                            "cue-conflict shape bias", "Task accuracy vs shape bias, by protocol"), "shape_bias_scatter")

    tc = task_correlations(df)
    tc.to_csv(OUT_TABLES / "task_correlations.csv")
    save(plot_corr_matrix(tc), "task_correlations")

    pd.set_option("display.width", 250)
    print(f"{len(df)} checkpoints, {df.shape_bias.notna().sum()} with shape bias")
    print("\nMadry - TRADES (mean over eps):\n", mmt.round(3).to_string(index=False))
    print("\nTRADES effect per task (raw / given clean / given clean + AA):\n",
          ctrl[["measure", "trades_effect_raw", "trades_effect_given_clean", "trades_se_given_clean",
                "trades_effect_given_clean_aa", "trades_se_given_clean_aa"]].round(3).to_string(index=False))
    print("\ntask correlations (selected):\n", tc.loc[["contrast", "uniform-noise", "high-pass", "shape_bias"]].round(2).T.to_string())
    print("\nshape-bias correlations:\n", sbc[["measure", "rho_all", "p_all", "partial_rho_given_clean",
                                              "rho_madry", "rho_trades"]].round(3).to_string(index=False))
    print(f"\nwrote {OUT_TABLES} and {OUT_PLOTS}")


if __name__ == "__main__":
    main()
