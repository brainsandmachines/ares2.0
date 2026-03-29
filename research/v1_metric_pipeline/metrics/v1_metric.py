from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.v1_metric_pipeline.models.v1_feature_extractor import V1FeatureExtractor
from research.v1_metric_pipeline.utils.config import V1MetricConfig


class V1PerceptualMetric(nn.Module):
    def __init__(self, extractor: V1FeatureExtractor, cfg: V1MetricConfig):
        super().__init__()
        self.extractor = extractor
        self.cfg = cfg

    def _normalized_l2(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        num = torch.mean((a - b) ** 2, dim=(2, 3))
        den = torch.mean((a ** 2 + b ** 2), dim=(2, 3)).clamp_min(self.cfg.eps)
        return torch.mean(num / den, dim=1)

    def _plain_l2(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mean((a - b) ** 2, dim=(1, 2, 3))

    def _maybe_pool(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.apply_pooling_before_compare:
            return a, b
        return F.avg_pool2d(a, kernel_size=2, stride=2), F.avg_pool2d(b, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor, reduction: Literal["mean", "sum", "none"] = "mean") -> torch.Tensor:
        fx = self.extractor(x)
        fxp = self.extractor(x_prime)
        simple_x, simple_xp = self._maybe_pool(fx["simple"], fxp["simple"])
        complex_x, complex_xp = self._maybe_pool(fx["complex"], fxp["complex"])

        if self.cfg.use_normalized_l2:
            ds = self._normalized_l2(simple_x, simple_xp)
            dc = self._normalized_l2(complex_x, complex_xp)
        else:
            ds = self._plain_l2(simple_x, simple_xp)
            dc = self._plain_l2(complex_x, complex_xp)

        d = self.cfg.alpha * ds + self.cfg.beta * dc

        if reduction == "mean":
            return d.mean()
        if reduction == "sum":
            return d.sum()
        if reduction == "none":
            return d
        raise ValueError(f"Unsupported reduction: {reduction}")
