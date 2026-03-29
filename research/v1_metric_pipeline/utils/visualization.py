from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def save_kernel_grid(kernels: torch.Tensor, out_path: str, max_kernels: int = 64, nrow: int = 8) -> None:
    # kernels shape: [F, C, K, K]
    k = kernels[:max_kernels].mean(dim=1, keepdim=True)
    k = (k - k.amin(dim=(1, 2, 3), keepdim=True)) / (
        k.amax(dim=(1, 2, 3), keepdim=True) - k.amin(dim=(1, 2, 3), keepdim=True) + 1e-8
    )
    grid = make_grid(k, nrow=nrow, normalize=False, pad_value=1.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


def save_feature_grid(feats: torch.Tensor, out_path: str, max_channels: int = 32, nrow: int = 8) -> None:
    # feats shape: [1, C, H, W] or [B, C, H, W]
    f = feats[0:1, :max_channels].transpose(0, 1)  # [C, 1, H, W]
    f = (f - f.amin(dim=(1, 2, 3), keepdim=True)) / (
        f.amax(dim=(1, 2, 3), keepdim=True) - f.amin(dim=(1, 2, 3), keepdim=True) + 1e-8
    )
    grid = make_grid(f, nrow=nrow, normalize=False, pad_value=1.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
