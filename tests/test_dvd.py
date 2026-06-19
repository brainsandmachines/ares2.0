import sys
import types

import torch
from omegaconf import OmegaConf

from ares.utils.dvd import (
    apply_dvd_to_batch,
    build_dvd_state,
    normalize_dvd_variant,
    resolve_months_per_epoch,
)


def test_dvd_variant_resolution():
    assert normalize_dvd_variant("dvd_p") == "dvd-p"
    assert normalize_dvd_variant("dvd-b") == "dvd-b"
    assert normalize_dvd_variant("s") == "dvd-s"
    assert resolve_months_per_epoch(OmegaConf.create({"variant": "dvd-p", "months_per_epoch": None})) == 4.0
    assert resolve_months_per_epoch(OmegaConf.create({"variant": "dvd-b", "months_per_epoch": 3.5})) == 3.5


def test_invalid_dvd_variant_rejected():
    try:
        normalize_dvd_variant("dvd-x")
    except ValueError as exc:
        assert "Unsupported DVD variant" in str(exc)
    else:
        raise AssertionError("invalid DVD variant should raise")


def test_build_dvd_state_uses_scale_free_backend(monkeypatch):
    calls = {}

    class _DVDConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _DVDTransformer:
        def __init__(self, config):
            calls["config"] = config

        def __call__(self, images, months):
            calls["months"] = months
            return images * 0.5

    def _generate_age_months_curve(total_epochs, len_train_loader, months_per_epoch, **kwargs):
        calls["curve_args"] = (total_epochs, len_train_loader, months_per_epoch, kwargs)
        return [0.0, 1.0, 2.0, 3.0]

    root = types.ModuleType("dvd_scale_free")
    package = types.ModuleType("dvd_scale_free.dvd_scale_free")
    development = types.ModuleType("dvd_scale_free.dvd_scale_free.development")
    development.DVDConfig = _DVDConfig
    development.DVDTransformer = _DVDTransformer
    development.generate_age_months_curve = _generate_age_months_curve
    monkeypatch.setitem(sys.modules, "dvd_scale_free", root)
    monkeypatch.setitem(sys.modules, "dvd_scale_free.dvd_scale_free", package)
    monkeypatch.setitem(sys.modules, "dvd_scale_free.dvd_scale_free.development", development)

    cfg = OmegaConf.create(
        {
            "seed": 7,
            "training": {"epochs": 2},
            "dataset": {
                "input_size": 224,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "dvd": {
                    "enabled": True,
                    "backend": "scale_free",
                    "variant": "dvd-b",
                    "months_per_epoch": None,
                    "time_order": "chronological",
                    "apply_blur": True,
                    "apply_color": True,
                    "apply_contrast": True,
                    "apply_threshold_color": False,
                    "cs_logspan_start": 0.005,
                },
            },
        }
    )

    state = build_dvd_state(cfg, len_train_loader=2)

    assert calls["curve_args"][0:3] == (2, 2, 2.0)
    assert calls["config"].cs_logspan_start == 0.005
    normalized = torch.zeros(1, 3, 2, 2)
    transformed = apply_dvd_to_batch(normalized, state, epoch=0, batch_idx=1, len_loader=2)
    assert calls["months"] == 1.0
    assert torch.allclose(transformed, torch.full_like(transformed, -0.5))
