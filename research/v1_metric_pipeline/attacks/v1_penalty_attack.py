from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from research.v1_metric_pipeline.metrics.v1_metric import V1PerceptualMetric
from research.v1_metric_pipeline.utils.config import V1PenaltyAttackConfig


class V1PenaltyAttack:
    def __init__(self, cfg: V1PenaltyAttackConfig):
        self.cfg = cfg

    def _clamp_to_guardrail(self, x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        if self.cfg.pixel_guardrail_eps is None:
            return torch.clamp(x_adv, 0.0, 1.0)
        lo = torch.clamp(x_orig - self.cfg.pixel_guardrail_eps, 0.0, 1.0)
        hi = torch.clamp(x_orig + self.cfg.pixel_guardrail_eps, 0.0, 1.0)
        return torch.max(torch.min(x_adv, hi), lo)

    def _shrink_back_if_needed(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
        metric: V1PerceptualMetric,
    ) -> torch.Tensor:
        if not self.cfg.shrink_back_enabled or self.cfg.v1_budget is None:
            return x_adv

        d = metric(x_orig, x_adv, reduction="none")
        if torch.all(d <= self.cfg.v1_budget):
            return x_adv

        out = x_adv.clone()
        for idx in range(x_adv.shape[0]):
            if d[idx] <= self.cfg.v1_budget:
                continue
            lo = 0.0
            hi = 1.0
            xi = x_orig[idx : idx + 1]
            xa = x_adv[idx : idx + 1]
            for _ in range(self.cfg.shrink_back_iters):
                t = 0.5 * (lo + hi)
                cand = xi + t * (xa - xi)
                dc = metric(xi, cand, reduction="none")[0]
                if dc <= self.cfg.v1_budget:
                    lo = t
                else:
                    hi = t
            out[idx : idx + 1] = xi + lo * (xa - xi)
        return out

    def generate(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        metric: V1PerceptualMetric,
    ) -> torch.Tensor:
        x_orig = x.detach()
        x_adv = x_orig.clone()

        if self.cfg.random_start:
            noise_eps = self.cfg.pixel_guardrail_eps if self.cfg.pixel_guardrail_eps is not None else self.cfg.random_start_eps
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-noise_eps, noise_eps)
            x_adv = self._clamp_to_guardrail(x_adv, x_orig)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.cfg.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            ce = F.cross_entropy(logits, y, reduction="mean")
            v1_pen = metric(x_orig, x_adv, reduction="mean")
            objective = ce - self.cfg.lambda_v1 * v1_pen
            grad = torch.autograd.grad(objective, x_adv, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                x_adv = x_adv + self.cfg.step_size * torch.sign(grad)
                x_adv = self._clamp_to_guardrail(x_adv, x_orig)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                x_adv = self._shrink_back_if_needed(x_adv, x_orig, metric)

            x_adv = x_adv.detach()

        return x_adv
