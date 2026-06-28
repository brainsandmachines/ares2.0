#!/usr/bin/env python3
"""Orchestrator plotting stage: AutoAttack best/last/advbest comparison + DB link.

A thin wrapper over ``plot_autoattack_comparation.py`` (reusing its heatmap and
CSV-loading logic unchanged) that additionally, for an orchestrated run:

1. picks the *best checkpoint* among (best, last, advbest) by the model's own
   threat model -- the AutoAttack ``robust_acc`` at the norm+eps encoded in its
   name (e.g. ``..._l2_2_init1`` -> highest l2 accuracy at eps 2). Baseline/clean
   models (no norm+eps) fall back to clean accuracy; and
2. writes the result back to the orchestrator DB: status -> FINISHED with the
   winning ``best_checkpoint`` column.

Run standalone exactly like the original (it just won't touch the DB unless the
ORCH_MODEL_ID / ORCH_DB env vars are set, which the controller exports).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

from ares.utils.epsilon_schedule import DEFAULT_EPS_INPUTS

# Reuse the original module wholesale (same-dir import when run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for orchestrator

from plot_autoattack_comparation import (  # noqa: E402
    CHECKPOINT_KINDS,
    accuracy_for_eval_eps,
    csv_name_for_checkpoint_kind,
    load_autoattack_results,
    model_entries,
    parse_args,
    plot_comparation,
)

# --------------------------------------------------------------------------
# Threat model (norm, eps) derivation from a model name.
# --------------------------------------------------------------------------
_EPS_RE_CONT = re.compile(r"_cont[0-9.]+to([0-9.]+)_(?:direct|ramp)_init[0-9]+$")
_EPS_RE_STD = re.compile(r"_([0-9]+(?:\.[0-9]+)?)_init[0-9]+$")

# The eps grid AutoAttack actually sweeps. A model whose nominal eps is not one
# of these has no matching column, so we score it at a representative sampled
# eps instead. And an adversarial model whose name carries no norm token (e.g.
# gradnorm_*) is scored under l2 by default.
AA_EPS = DEFAULT_EPS_INPUTS
DEFAULT_EPS = 4.0
DEFAULT_NORM = "l2"


def norm_from_name(model_name: str) -> str | None:
    """Return 'linf' | 'l2' | 'l1' encoded in the name, or None (baseline/clean)."""
    if "linf" in model_name:
        return "linf"
    if "l2" in model_name:
        return "l2"
    if "l1" in model_name:
        return "l1"
    return None


def eps_from_name(model_name: str) -> float | None:
    m = _EPS_RE_CONT.search(model_name)
    if m:
        return float(m.group(1))
    m = _EPS_RE_STD.search(model_name)
    if m:
        return float(m.group(1))
    return None


def best_checkpoint_kind(models_root: Path, model_name: str) -> tuple[str | None, float | None]:
    """Pick the best checkpoint kind AND its score for the model's threat model.

    Score is the AutoAttack ``robust_acc`` at the model's norm+eps (e.g.
    ``..._l2_2_init1`` -> l2 accuracy at eps 2). Defaults: an adversarial model
    whose name has no norm token (e.g. ``gradnorm_*``) is scored under
    ``l2``, and an eps that is not on the swept grid (``AA_EPS``) snaps to
    ``DEFAULT_EPS`` (4). Baseline/clean models (no eps in the name) fall back to
    clean accuracy. Returns ``(kind, score)`` or ``(None, None)`` if no CSVs
    exist. The score (in %) is what gets written to the DB alongside the kind.
    """
    norm = norm_from_name(model_name)
    eps = eps_from_name(model_name)
    # No eps in the name -> clean-trained (baseline/clean): score by clean acc.
    # Otherwise it's an adversarial model: default a missing norm to l2 (e.g.
    # gradnorm_*) and snap an off-grid eps to a sampled AA eps.
    use_robust = eps is not None and eps > 0
    if use_robust:
        if norm is None:
            norm = DEFAULT_NORM
        if not any(abs(float(eps) - e) < 1e-9 for e in AA_EPS):
            eps = DEFAULT_EPS

    best_kind: str | None = None
    best_val: float | None = None
    for kind in CHECKPOINT_KINDS:
        csv_path = models_root / model_name / csv_name_for_checkpoint_kind(kind)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "clean_acc" not in df.columns:
            continue
        df["attack_norm"] = df.get("attack_norm", "").astype(str).str.lower()
        try:
            if use_robust:
                df_norm = df[df["attack_norm"] == norm].copy()
                if df_norm.empty:
                    continue
                val = accuracy_for_eval_eps(df_norm, csv_path, norm, float(eps))
            else:
                clean = pd.to_numeric(df["clean_acc"], errors="coerce").dropna()
                if clean.empty:
                    continue
                val = float(clean.mean())
        except (ValueError, KeyError):
            continue
        if best_val is None or val > best_val:
            best_val, best_kind = val, kind

    metric = f"{norm} robust@eps{eps:g}" if use_robust else "clean"
    print(f"[orch-plot] best checkpoint for {model_name}: {best_kind} "
          f"({metric}={best_val if best_kind else 'n/a'})")
    return best_kind, best_val


def _finish_in_db(best_kind: str | None, best_score: float | None) -> None:
    """Write status FINISHED + best_checkpoint + best_score to the DB (if tracked)."""
    model_id = os.environ.get("ORCH_MODEL_ID")
    db_path = os.environ.get("ORCH_DB")
    if not model_id or not db_path:
        print("[orch-plot] not orchestrated (ORCH_MODEL_ID/ORCH_DB unset); skipping DB link")
        return
    from orchestrator.db import OrchestratorDB

    OrchestratorDB(db_path).mark_finished(model_id, best_kind, best_score)
    print(f"[orch-plot] DB: {model_id} -> FINISHED "
          f"(best_checkpoint={best_kind}, best_score={best_score})")


def main() -> None:
    args = parse_args()
    out_png = args.out_dir / f"autoattack_eval_comparation_{args.plot_name}.png"
    out_svg = args.out_dir / f"autoattack_eval_comparation_{args.plot_name}.svg" if args.svg else None
    entries = model_entries(args.models_root, args.models, args.checkpoint_kind)
    data_by_norm = load_autoattack_results(args.models_root, entries, args.eval_eps_order)
    model_labels = [label for _model_name, label, _csv_name in entries]
    plot_comparation(
        data_by_norm, model_labels, args.eval_eps_order, args.plot_name,
        out_png, out_svg=out_svg, show=args.show,
    )
    print(f"Saved AutoAttack comparation plot to {out_png}")

    # Orchestrator handoff: pick the winning checkpoint + score, mark FINISHED.
    # The single-model run is the orchestrated case; --models[0] is its dir name.
    best_kind, best_score = best_checkpoint_kind(Path(args.models_root), args.models[0])
    _finish_in_db(best_kind, best_score)


if __name__ == "__main__":
    main()
