from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.v1_metric_pipeline.utils.config import V1ExtractorConfig


@dataclass
class V1FeatureOutput:
    simple: torch.Tensor
    complex: torch.Tensor
    all_linear: torch.Tensor


def _make_orientations(num_orientations: int) -> torch.Tensor:
    return torch.linspace(0.0, math.pi, steps=num_orientations + 1)[:-1]


def _sample_spatial_frequencies_cpd(cfg: V1ExtractorConfig) -> torch.Tensor:
    freq_cfg = cfg.frequency_sampling
    if freq_cfg.policy == "explicit":
        if not freq_cfg.explicit_cpd:
            raise ValueError("frequency_sampling.policy='explicit' requires explicit_cpd values")
        return torch.tensor(freq_cfg.explicit_cpd, dtype=torch.float32)
    if freq_cfg.policy == "linear":
        return torch.linspace(freq_cfg.min_cpd, freq_cfg.max_cpd, steps=freq_cfg.num_frequencies)
    if freq_cfg.policy == "logspace":
        return torch.logspace(
            math.log10(freq_cfg.min_cpd),
            math.log10(freq_cfg.max_cpd),
            steps=freq_cfg.num_frequencies,
        )
    raise ValueError(f"Unknown frequency sampling policy: {freq_cfg.policy}")


def cpd_to_cpp(cycles_per_degree: torch.Tensor, image_size_px: int, field_of_view_degrees: float) -> torch.Tensor:
    degrees_per_pixel = field_of_view_degrees / float(image_size_px)
    return cycles_per_degree * degrees_per_pixel


def _build_gabor_kernel(
    kernel_size: int,
    wavelength_px: float,
    orientation_rad: float,
    phase_rad: float,
    aspect_ratio: float,
    sigma_scale: float,
) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    cos_t = math.cos(orientation_rad)
    sin_t = math.sin(orientation_rad)
    x_theta = xx * cos_t + yy * sin_t
    y_theta = -xx * sin_t + yy * cos_t

    sigma = max(1.0, sigma_scale * wavelength_px)
    env = torch.exp(-(x_theta ** 2 + (aspect_ratio ** 2) * (y_theta ** 2)) / (2.0 * sigma ** 2))
    carrier = torch.cos((2.0 * math.pi / max(wavelength_px, 1e-6)) * x_theta + phase_rad)
    kernel = env * carrier
    kernel = kernel - kernel.mean()
    kernel = kernel / (kernel.norm(p=2) + 1e-8)
    return kernel


class V1FeatureExtractor(nn.Module):
    """Deterministic V1-like front-end with fixed Gabor filters and simple/complex nonlinearities."""

    def __init__(self, cfg: V1ExtractorConfig):
        super().__init__()
        if len(cfg.phases) != 2:
            raise ValueError("This implementation expects 2 phases (quadrature pair), e.g., [0, pi/2].")
        self.cfg = cfg

        kernels, pair_indices = self._build_filter_bank(cfg)
        self.register_buffer("gabor_kernels", kernels)

        self.simple_indices = self._select_simple_indices(cfg, pair_indices)
        self.complex_pair_indices = self._select_complex_pairs(cfg, pair_indices)

        self.register_buffer("simple_mean", torch.zeros(len(self.simple_indices), dtype=torch.float32))
        self.register_buffer("simple_std", torch.ones(len(self.simple_indices), dtype=torch.float32))
        self.register_buffer("complex_mean", torch.zeros(len(self.complex_pair_indices), dtype=torch.float32))
        self.register_buffer("complex_std", torch.ones(len(self.complex_pair_indices), dtype=torch.float32))

    @staticmethod
    def _select_simple_indices(cfg: V1ExtractorConfig, pair_indices: List[Tuple[int, int]]) -> List[int]:
        all_indices = [idx for pair in pair_indices for idx in pair]
        if cfg.num_simple_channels is None:
            return all_indices
        return all_indices[: cfg.num_simple_channels]

    @staticmethod
    def _select_complex_pairs(cfg: V1ExtractorConfig, pair_indices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if cfg.num_complex_channels is None:
            return pair_indices
        return pair_indices[: cfg.num_complex_channels]

    @staticmethod
    def _build_filter_bank(cfg: V1ExtractorConfig) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        orientations = _make_orientations(cfg.num_orientations)
        cpd = _sample_spatial_frequencies_cpd(cfg)
        cpp = cpd_to_cpp(cpd, cfg.image_size_px, cfg.field_of_view_degrees)
        wavelengths = 1.0 / torch.clamp(cpp, min=1e-6)

        all_kernels: List[torch.Tensor] = []
        pair_indices: List[Tuple[int, int]] = []

        for orientation in orientations.tolist():
            for wavelength_px in wavelengths.tolist():
                idx0 = len(all_kernels)
                k0 = _build_gabor_kernel(
                    kernel_size=cfg.kernel_size,
                    wavelength_px=wavelength_px,
                    orientation_rad=orientation,
                    phase_rad=cfg.phases[0],
                    aspect_ratio=cfg.aspect_ratio,
                    sigma_scale=cfg.sigma_scale,
                )
                all_kernels.append(k0)

                idx1 = len(all_kernels)
                k1 = _build_gabor_kernel(
                    kernel_size=cfg.kernel_size,
                    wavelength_px=wavelength_px,
                    orientation_rad=orientation,
                    phase_rad=cfg.phases[1],
                    aspect_ratio=cfg.aspect_ratio,
                    sigma_scale=cfg.sigma_scale,
                )
                all_kernels.append(k1)
                pair_indices.append((idx0, idx1))

        stacked = torch.stack(all_kernels, dim=0)  # [F, K, K]
        # Replicate over input channels with equal weighting.
        stacked = stacked[:, None, :, :].repeat(1, cfg.in_channels, 1, 1) / float(cfg.in_channels)
        return stacked, pair_indices

    def set_channel_stats(
        self,
        simple_mean: Optional[torch.Tensor] = None,
        simple_std: Optional[torch.Tensor] = None,
        complex_mean: Optional[torch.Tensor] = None,
        complex_std: Optional[torch.Tensor] = None,
    ) -> None:
        if simple_mean is not None:
            self.simple_mean.copy_(simple_mean.to(self.simple_mean.device, self.simple_mean.dtype))
        if simple_std is not None:
            self.simple_std.copy_(simple_std.to(self.simple_std.device, self.simple_std.dtype))
        if complex_mean is not None:
            self.complex_mean.copy_(complex_mean.to(self.complex_mean.device, self.complex_mean.dtype))
        if complex_std is not None:
            self.complex_std.copy_(complex_std.to(self.complex_std.device, self.complex_std.dtype))

    def get_gabor_kernels(self) -> torch.Tensor:
        return self.gabor_kernels.detach().clone()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        responses = F.conv2d(
            x,
            weight=self.gabor_kernels,
            bias=None,
            stride=self.cfg.stride,
            padding=self.cfg.kernel_size // 2,
        )

        simple_linear = responses[:, self.simple_indices, :, :]
        simple = F.relu(simple_linear)

        idx0 = [p[0] for p in self.complex_pair_indices]
        idx1 = [p[1] for p in self.complex_pair_indices]
        r0 = responses[:, idx0, :, :]
        r1 = responses[:, idx1, :, :]
        complex_map = torch.sqrt(r0.pow(2) + r1.pow(2) + 1e-8)

        if self.cfg.use_pooling:
            simple = F.avg_pool2d(simple, kernel_size=self.cfg.pooling_kernel, stride=self.cfg.pooling_stride)
            complex_map = F.avg_pool2d(complex_map, kernel_size=self.cfg.pooling_kernel, stride=self.cfg.pooling_stride)

        if self.cfg.use_channel_stats_norm:
            simple = (simple - self.simple_mean[None, :, None, None]) / (self.simple_std[None, :, None, None] + 1e-6)
            complex_map = (complex_map - self.complex_mean[None, :, None, None]) / (
                self.complex_std[None, :, None, None] + 1e-6
            )

        return {
            "simple": simple,
            "complex": complex_map,
            "all_linear": responses,
        }
