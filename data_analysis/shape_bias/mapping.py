"""ImageNet-1K to 16 cue-conflict category mapping.

Vendored verbatim from the shape-bias-analysis repo
(shape_bias_analysis/mapping.py). The category set and ImageNet index lists follow
rgeirhos/texture-vs-shape commit ccd0c01d42c0a8a834e90ec141e7f24fa915dc23.
"""

from __future__ import annotations

import torch


CATEGORIES = [
    "airplane",
    "bear",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "car",
    "cat",
    "chair",
    "clock",
    "dog",
    "elephant",
    "keyboard",
    "knife",
    "oven",
    "truck",
]

CATEGORY_TO_INDICES = {
    "airplane": [404],
    "bear": [294, 295, 296, 297],
    "bicycle": [444, 671],
    "bird": [
        8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23,
        24, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93,
        94, 95, 96, 98, 99, 100, 127, 128, 129, 130, 131,
        132, 133, 135, 136, 137, 138, 139, 140, 141, 142,
        143, 144, 145,
    ],
    "boat": [472, 554, 625, 814, 914],
    "bottle": [440, 720, 737, 898, 899, 901, 907],
    "car": [436, 511, 817],
    "cat": [281, 282, 283, 284, 285, 286],
    "chair": [423, 559, 765, 857],
    "clock": [409, 530, 892],
    "dog": [
        152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
        162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
        172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
        182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
        193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
        203, 205, 206, 207, 208, 209, 210, 211, 212, 213,
        214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 228, 229, 230, 231, 232, 233, 234,
        235, 236, 237, 238, 239, 240, 241, 243, 244, 245,
        246, 247, 248, 249, 250, 252, 253, 254, 255, 256,
        257, 259, 261, 262, 263, 265, 266, 267, 268,
    ],
    "elephant": [385, 386],
    "keyboard": [508, 878],
    "knife": [499],
    "oven": [766],
    "truck": [555, 569, 656, 675, 717, 734, 864, 867],
}


def aggregate_to_16(probabilities: torch.Tensor, aggregation: str = "mean") -> torch.Tensor:
    """Aggregate ImageNet-1K probabilities into the 16 cue-conflict categories."""
    if probabilities.ndim != 2 or probabilities.shape[1] != 1000:
        raise ValueError(f"Expected probabilities with shape [batch, 1000], got {tuple(probabilities.shape)}")
    if aggregation not in {"mean", "max"}:
        raise ValueError(f"Unsupported aggregation {aggregation!r}; expected 'mean' or 'max'")

    category_scores = []
    for category in CATEGORIES:
        indices = torch.as_tensor(CATEGORY_TO_INDICES[category], device=probabilities.device)
        values = probabilities.index_select(1, indices)
        if aggregation == "mean":
            category_scores.append(values.mean(dim=1))
        else:
            category_scores.append(values.max(dim=1).values)
    return torch.stack(category_scores, dim=1)


def decisions_from_scores(category_scores: torch.Tensor) -> list[str]:
    """Return the max-scoring 16-way category decision for each row."""
    if category_scores.ndim != 2 or category_scores.shape[1] != len(CATEGORIES):
        raise ValueError(
            f"Expected category scores with shape [batch, {len(CATEGORIES)}], "
            f"got {tuple(category_scores.shape)}"
        )
    return [CATEGORIES[int(i)] for i in category_scores.argmax(dim=1).tolist()]


def category_index(category: str) -> int:
    return CATEGORIES.index(category)
