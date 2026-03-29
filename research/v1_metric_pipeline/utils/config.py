from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FrequencySamplingConfig:
    policy: str = "logspace"  # logspace | linear | explicit
    min_cpd: float = 0.5
    max_cpd: float = 12.0
    num_frequencies: int = 6
    explicit_cpd: List[float] = field(default_factory=list)


@dataclass
class V1ExtractorConfig:
    image_size_px: int = 224
    field_of_view_degrees: float = 8.0
    in_channels: int = 3
    kernel_size: int = 25
    stride: int = 4
    num_orientations: int = 8
    phases: List[float] = field(default_factory=lambda: [0.0, 1.57079632679])
    aspect_ratio: float = 0.5
    sigma_scale: float = 0.56
    use_pooling: bool = True
    pooling_kernel: int = 2
    pooling_stride: int = 2
    num_simple_channels: Optional[int] = None
    num_complex_channels: Optional[int] = None
    use_channel_stats_norm: bool = False
    frequency_sampling: FrequencySamplingConfig = field(default_factory=FrequencySamplingConfig)


@dataclass
class V1MetricConfig:
    alpha: float = 0.3
    beta: float = 0.7
    eps: float = 1e-8
    apply_pooling_before_compare: bool = False
    use_normalized_l2: bool = True


@dataclass
class V1PenaltyAttackConfig:
    steps: int = 20
    step_size: float = 0.005
    lambda_v1: float = 2.0
    random_start: bool = True
    random_start_eps: float = 0.01
    pixel_guardrail_eps: Optional[float] = 0.03
    shrink_back_enabled: bool = False
    v1_budget: Optional[float] = None
    shrink_back_iters: int = 5


@dataclass
class DemoConfig:
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "research/v1_metric_pipeline/outputs/demo"
    model_num_classes: int = 10


@dataclass
class TrainScaffoldConfig:
    seed: int = 0
    device: str = "cpu"
    batch_size: int = 8
    image_size_px: int = 224
    num_classes: int = 10
    dataset_size: int = 128
    epochs: int = 1
    max_batches_per_epoch: int = 10
    lr: float = 1e-3


@dataclass
class PipelineConfig:
    extractor: V1ExtractorConfig = field(default_factory=V1ExtractorConfig)
    metric: V1MetricConfig = field(default_factory=V1MetricConfig)
    attack: V1PenaltyAttackConfig = field(default_factory=V1PenaltyAttackConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    train: TrainScaffoldConfig = field(default_factory=TrainScaffoldConfig)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _from_dict(dataclass_type, data: Dict[str, Any]):
    field_names = {f.name for f in dataclass_type.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {}
    for name in field_names:
        if name not in data:
            continue
        value = data[name]
        field_type = dataclass_type.__dataclass_fields__[name].type  # type: ignore[attr-defined]
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[name] = _from_dict(field_type, value)
        else:
            kwargs[name] = value
    return dataclass_type(**kwargs)


def load_pipeline_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    base = asdict(PipelineConfig())
    if config_path is not None:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Top-level YAML must be a mapping, got: {type(loaded)}")
        base = _deep_update(base, loaded)
    if overrides:
        base = _deep_update(base, overrides)

    cfg = PipelineConfig(
        extractor=_from_dict(V1ExtractorConfig, base.get("extractor", {})),
        metric=_from_dict(V1MetricConfig, base.get("metric", {})),
        attack=_from_dict(V1PenaltyAttackConfig, base.get("attack", {})),
        demo=_from_dict(DemoConfig, base.get("demo", {})),
        train=_from_dict(TrainScaffoldConfig, base.get("train", {})),
    )
    cfg.extractor.frequency_sampling = _from_dict(
        FrequencySamplingConfig,
        base.get("extractor", {}).get("frequency_sampling", {}),
    )
    return cfg


def flatten_dataclass(dc: Any) -> Dict[str, Any]:
    return asdict(dc)
