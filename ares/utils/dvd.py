import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


DVD_MONTHS_PER_EPOCH = {
    "dvd-p": 4.0,
    "dvd_p": 4.0,
    "p": 4.0,
    "dvd-b": 2.0,
    "dvd_b": 2.0,
    "b": 2.0,
    "dvd-s": 1.0,
    "dvd_s": 1.0,
    "s": 1.0,
}


@dataclass
class DVDState:
    transformer: Callable[[torch.Tensor, float], torch.Tensor]
    age_months_curve: list[float]
    mean: torch.Tensor
    std: torch.Tensor


def normalize_dvd_variant(variant: str) -> str:
    key = str(variant).strip().lower()
    if key not in DVD_MONTHS_PER_EPOCH:
        supported = ", ".join(("dvd-p", "dvd-b", "dvd-s"))
        raise ValueError(f"Unsupported DVD variant '{variant}'. Supported variants: {supported}")
    if key in {"p", "dvd_p", "dvd-p"}:
        return "dvd-p"
    if key in {"b", "dvd_b", "dvd-b"}:
        return "dvd-b"
    return "dvd-s"


def resolve_months_per_epoch(dvd_cfg: Any) -> float:
    explicit = dvd_cfg.get("months_per_epoch", None)
    if explicit is not None:
        return float(explicit)
    return DVD_MONTHS_PER_EPOCH[normalize_dvd_variant(dvd_cfg.get("variant", "dvd-b"))]


def _import_dvd_backend(backend: str):
    backend = str(backend or "paper").lower()
    if backend == "scale_free":
        try:
            from dvd_scale_free.dvd_scale_free.development import (  # type: ignore
                DVDConfig,
                DVDTransformer,
                generate_age_months_curve,
            )

            return DVDConfig, DVDTransformer, generate_age_months_curve
        except ImportError as exc:
            raise ImportError(
                "dataset.dvd.enabled=true requires the official DVD scale-free package "
                "and kornia. Install KietzmannLab/DVD scale_free branch in the training env."
            ) from exc
    if backend == "paper":
        try:
            from dvd.dvd.development import DVDConfig, DVDTransformer, generate_age_months_curve  # type: ignore

            return DVDConfig, DVDTransformer, generate_age_months_curve
        except ImportError as exc:
            raise ImportError(
                "dataset.dvd.backend=paper requires the official DVD paper branch package and kornia."
            ) from exc
    raise ValueError(f"Unsupported DVD backend: {backend}")


def _make_age_curve(generate_age_months_curve, cfg: Any, len_train_loader: int) -> list[float]:
    dvd_cfg = cfg.dataset.dvd
    months_per_epoch = resolve_months_per_epoch(dvd_cfg)
    time_order = str(dvd_cfg.get("time_order", "chronological")).lower()
    total_epochs = int(cfg.training.epochs)
    total_steps = total_epochs * int(len_train_loader)

    if time_order == "fully_random":
        rng = random.Random(None if cfg.seed is None else int(cfg.seed))
        max_age = months_per_epoch * total_epochs
        return [rng.random() * max_age for _ in range(total_steps)]

    return list(
        generate_age_months_curve(
            total_epochs,
            len_train_loader,
            months_per_epoch,
            mid_phase=time_order == "mid_phase",
            shuffle=time_order == "random",
            seed=None if cfg.seed is None else int(cfg.seed),
        )
    )


def build_dvd_state(cfg: Any, len_train_loader: int) -> Optional[DVDState]:
    dvd_cfg = cfg.dataset.get("dvd", None)
    if dvd_cfg is None or not bool(dvd_cfg.get("enabled", False)):
        return None

    backend = str(dvd_cfg.get("backend", "paper")).lower()
    DVDConfig, DVDTransformer, generate_age_months_curve = _import_dvd_backend(backend)
    image_size = int(dvd_cfg.get("image_size", None) or cfg.dataset.input_size)
    config_kwargs = dict(
        apply_blur=int(bool(dvd_cfg.get("apply_blur", True))),
        apply_color=int(bool(dvd_cfg.get("apply_color", True))),
        apply_contrast=int(bool(dvd_cfg.get("apply_contrast", True))),
        apply_threshold_color=bool(dvd_cfg.get("apply_threshold_color", False)),
        image_size=image_size,
        fully_random=str(dvd_cfg.get("time_order", "chronological")).lower() == "fully_random",
    )
    if backend == "scale_free":
        config_kwargs["cs_logspan_start"] = float(dvd_cfg.get("cs_logspan_start", 5e-3))
    else:
        config_kwargs["contrast_amplitude_beta"] = float(dvd_cfg.get("contrast_amplitude_beta", 1e-4))
        config_kwargs["contrast_amplitude_lam"] = float(dvd_cfg.get("contrast_amplitude_lam", 150.0))
    config = DVDConfig(**config_kwargs)
    age_months_curve = _make_age_curve(generate_age_months_curve, cfg, len_train_loader)
    config.age_months_curve = age_months_curve
    transformer = DVDTransformer(config)
    mean = torch.tensor(cfg.dataset.mean, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(cfg.dataset.std, dtype=torch.float32).view(1, -1, 1, 1)
    return DVDState(transformer=transformer, age_months_curve=age_months_curve, mean=mean, std=std)


def apply_dvd_to_batch(input_tensor: torch.Tensor, dvd_state: DVDState, epoch: int, batch_idx: int, len_loader: int) -> torch.Tensor:
    global_step = int(epoch) * int(len_loader) + int(batch_idx)
    if global_step >= len(dvd_state.age_months_curve):
        raise IndexError(
            f"DVD age curve too short: global_step={global_step}, len={len(dvd_state.age_months_curve)}"
        )
    mean = dvd_state.mean.to(device=input_tensor.device, dtype=input_tensor.dtype)
    std = dvd_state.std.to(device=input_tensor.device, dtype=input_tensor.dtype)
    images_01 = (input_tensor * std + mean).clamp(0, 1)
    aged_01 = dvd_state.transformer(images_01, float(dvd_state.age_months_curve[global_step]))
    return (aged_01.clamp(0, 1) - mean) / std
