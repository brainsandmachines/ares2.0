from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image


def load_image_as_tensor(path: str, image_size_px: int) -> torch.Tensor:
    img = read_image(path).float() / 255.0
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] > 3:
        img = img[:3]
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=(image_size_px, image_size_px), mode="bilinear", align_corners=False)
    return img


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.detach().cpu().clamp(0.0, 1.0), str(out_path))
