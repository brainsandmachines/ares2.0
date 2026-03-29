import torch

from research.v1_metric_pipeline.attacks.v1_penalty_attack import V1PenaltyAttack
from research.v1_metric_pipeline.metrics.v1_metric import V1PerceptualMetric
from research.v1_metric_pipeline.models.tiny_cnn import TinyCNN
from research.v1_metric_pipeline.models.v1_feature_extractor import V1FeatureExtractor
from research.v1_metric_pipeline.utils.config import V1ExtractorConfig, V1MetricConfig, V1PenaltyAttackConfig


def test_attack_respects_box_and_guardrail() -> None:
    torch.manual_seed(0)
    x = torch.rand(4, 3, 64, 64)
    y = torch.randint(low=0, high=10, size=(4,))

    model = TinyCNN(in_channels=3, num_classes=10)
    extractor = V1FeatureExtractor(V1ExtractorConfig(image_size_px=64, kernel_size=15, stride=2))
    metric = V1PerceptualMetric(extractor, V1MetricConfig())
    attack = V1PenaltyAttack(
        V1PenaltyAttackConfig(
            steps=5,
            step_size=0.01,
            lambda_v1=1.0,
            random_start=True,
            pixel_guardrail_eps=0.05,
        )
    )

    x_adv = attack.generate(model=model, x=x, y=y, metric=metric)
    assert torch.all(x_adv >= 0.0)
    assert torch.all(x_adv <= 1.0)
    assert torch.max(torch.abs(x_adv - x)) <= 0.051


def test_attack_shrink_back_budget() -> None:
    torch.manual_seed(1)
    x = torch.rand(2, 3, 64, 64)
    y = torch.randint(low=0, high=10, size=(2,))

    model = TinyCNN(in_channels=3, num_classes=10)
    extractor = V1FeatureExtractor(V1ExtractorConfig(image_size_px=64, kernel_size=15, stride=2))
    metric = V1PerceptualMetric(extractor, V1MetricConfig())

    attack = V1PenaltyAttack(
        V1PenaltyAttackConfig(
            steps=3,
            step_size=0.02,
            lambda_v1=0.5,
            random_start=False,
            pixel_guardrail_eps=0.2,
            shrink_back_enabled=True,
            v1_budget=0.02,
            shrink_back_iters=4,
        )
    )

    x_adv = attack.generate(model=model, x=x, y=y, metric=metric)
    d = metric(x, x_adv, reduction="none")
    assert torch.all(d <= 0.021)
