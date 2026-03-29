import torch

from research.v1_metric_pipeline.metrics.v1_metric import V1PerceptualMetric
from research.v1_metric_pipeline.models.v1_feature_extractor import V1FeatureExtractor
from research.v1_metric_pipeline.utils.config import V1ExtractorConfig, V1MetricConfig


def test_metric_identity_and_symmetry() -> None:
    extractor = V1FeatureExtractor(V1ExtractorConfig(image_size_px=64, kernel_size=15, stride=2))
    metric = V1PerceptualMetric(extractor, V1MetricConfig(alpha=0.3, beta=0.7))

    x = torch.rand(3, 3, 64, 64)
    y = torch.rand(3, 3, 64, 64)

    d_xx = metric(x, x, reduction="none")
    d_xy = metric(x, y, reduction="none")
    d_yx = metric(y, x, reduction="none")

    assert torch.all(d_xx < 1e-7)
    assert torch.allclose(d_xy, d_yx, atol=1e-6)


def test_metric_backward() -> None:
    extractor = V1FeatureExtractor(V1ExtractorConfig(image_size_px=64, kernel_size=15, stride=2))
    metric = V1PerceptualMetric(extractor, V1MetricConfig())

    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    y = torch.rand(2, 3, 64, 64)

    loss = metric(x, y, reduction="mean")
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
