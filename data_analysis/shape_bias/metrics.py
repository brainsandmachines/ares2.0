"""Shape-bias counting helpers.

Vendored verbatim from the shape-bias-analysis repo (shape_bias_analysis/metrics.py).
"""

from __future__ import annotations


def classify_decision(shape_label: str, texture_label: str, decision: str) -> tuple[str, bool]:
    """Classify one cue-conflict decision.

    Returns `(outcome, contributes_to_denominator)`.
    """
    if shape_label == texture_label:
        return "excluded_same_category", False
    if decision == shape_label:
        return "shape", True
    if decision == texture_label:
        return "texture", True
    return "other", False


def summarize_image_rows(rows: list[dict]) -> dict:
    n_total = len(rows)
    n_conflict = sum(1 for row in rows if row["shape_label"] != row["texture_label"])
    n_shape = sum(1 for row in rows if row["outcome"] == "shape")
    n_texture = sum(1 for row in rows if row["outcome"] == "texture")
    n_other = sum(1 for row in rows if row["outcome"] == "other")
    denominator = n_shape + n_texture
    shape_bias = n_shape / denominator if denominator else float("nan")
    return {
        "n_total": n_total,
        "n_conflict": n_conflict,
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_other,
        "denominator": denominator,
        "shape_bias": shape_bias,
    }
