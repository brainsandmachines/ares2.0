"""Pick the winning checkpoint kind (best/last/advbest) by a model's threat model.

Used by the in-training ``write_best_checkpoint`` hook (the training run already
produced the AutoAttack sweep CSVs via ``final_eval``). The threat model
(norm + eps) is passed explicitly (from the CSV's ``threat_norm``/``threat_eps``),
so the non-standard AIRCC folder names parse correctly. Adversarial rows score by
AutoAttack ``robust_acc`` at (norm, eps); baseline / dvd_b_baseline rows (no eps)
score by mean clean accuracy.

Reuses the scoring helpers from ``data_analysis/plot_autoattack_comparation.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

CKPT_FILE_FOR_KIND = {
    "best": "model_best.pth.tar",
    "last": "last.pth.tar",
    "advbest": "model_best_adv.pth.tar",
}
AA_EPS_GRID = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0)


def best_checkpoint_for_threat(
    model_dir: Path, norm: Optional[str], eps: Optional[float]
) -> tuple[Optional[str], Optional[float]]:
    """Return (absolute checkpoint path, score) of the winning kind, or (None, None).

    ``norm``/``eps`` empty -> score by clean accuracy (baselines). Otherwise score
    by robust_acc at (norm, eps).
    """
    import pandas as pd  # heavy, import lazily

    for p in (str(REPO_ROOT), str(REPO_ROOT / "data_analysis")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from plot_autoattack_comparation import (  # noqa: E402
        CHECKPOINT_KINDS, accuracy_for_eval_eps, csv_name_for_checkpoint_kind,
    )

    use_robust = bool(norm) and eps is not None and float(eps) > 0
    norm = (norm or "").strip().lower()
    if use_robust:
        eps = float(eps)
        if eps not in AA_EPS_GRID:
            eps = 4.0  # off-grid safety (should not happen for our eps set)

    best_kind: Optional[str] = None
    best_val: Optional[float] = None
    for kind in CHECKPOINT_KINDS:
        csv_path = model_dir / csv_name_for_checkpoint_kind(kind)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "clean_acc" not in df.columns:
            continue
        try:
            if use_robust:
                df["attack_norm"] = df.get("attack_norm", "").astype(str).str.lower()
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

    if best_kind is None:
        return None, None
    return str((model_dir / CKPT_FILE_FOR_KIND[best_kind]).resolve()), best_val
