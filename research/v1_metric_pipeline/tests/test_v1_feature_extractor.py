import torch

from research.v1_metric_pipeline.models.v1_feature_extractor import V1FeatureExtractor
from research.v1_metric_pipeline.utils.config import V1ExtractorConfig


def test_feature_extractor_shapes_and_determinism() -> None:
    cfg = V1ExtractorConfig(
        image_size_px=64,
        kernel_size=15,
        stride=2,
        use_pooling=True,
        pooling_kernel=2,
        pooling_stride=2,
    )
    model = V1FeatureExtractor(cfg)
    x = torch.rand(2, 3, 64, 64)
    out1 = model(x)
    out2 = model(x)

    assert out1["simple"].shape == out2["simple"].shape
    assert out1["complex"].shape == out2["complex"].shape
    assert torch.allclose(out1["simple"], out2["simple"], atol=1e-6)
    assert torch.allclose(out1["complex"], out2["complex"], atol=1e-6)
    assert out1["simple"].shape[0] == 2
    assert out1["complex"].shape[0] == 2
