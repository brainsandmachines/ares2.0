"""Cue-conflict stimulus dataset.

Vendored from the shape-bias-analysis repo (shape_bias_analysis/dataset.py). The only
change from the original is the relative import of ``CATEGORIES`` below.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .mapping import CATEGORIES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def category_from_token(token: str) -> str:
    match = re.match(r"([a-zA-Z]+)", token)
    if not match:
        raise ValueError(f"Cannot parse category from token: {token!r}")
    category = match.group(1).lower()
    if category not in CATEGORIES:
        raise ValueError(f"Unknown cue-conflict category {category!r} from token {token!r}")
    return category


def parse_stimulus_path(path: str | Path, stimuli_root: str | Path | None = None) -> dict:
    path = Path(path)
    shape_label = path.parent.name.lower()
    if shape_label not in CATEGORIES:
        raise ValueError(f"Unknown shape category folder: {path.parent.name!r}")

    parts = path.stem.split("-")
    if len(parts) < 2:
        raise ValueError(f"Expected filename like shapeN-textureM.png, got: {path.name}")
    filename_shape = category_from_token(parts[0])
    texture_label = category_from_token(parts[1])
    if filename_shape != shape_label:
        raise ValueError(
            f"Folder shape {shape_label!r} does not match filename shape {filename_shape!r}: {path}"
        )

    rel_path = str(path.relative_to(stimuli_root)) if stimuli_root else str(path)
    return {
        "image_path": str(path),
        "relative_path": rel_path,
        "shape_label": shape_label,
        "texture_label": texture_label,
    }


def default_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


class CueConflictDataset(Dataset):
    def __init__(self, stimuli_root: str | Path, transform=None):
        self.stimuli_root = Path(stimuli_root)
        if not self.stimuli_root.exists():
            raise FileNotFoundError(f"Stimuli root does not exist: {self.stimuli_root}")
        self.transform = transform or default_transform()
        self.samples = self._discover_samples()
        if not self.samples:
            raise ValueError(f"No stimulus images found under {self.stimuli_root}")

    def _discover_samples(self) -> list[dict]:
        samples = []
        for path in sorted(self.stimuli_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(parse_stimulus_path(path, self.stimuli_root))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample
