import torch

from ares.model.v1_convnext import V1ConvNeXt


def test_v1_convnext_feature_interface_shapes():
    model = V1ConvNeXt(
        backbone_name="convnext_small",
        input_size=224,
        noise_mode=None,
    )
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        v1 = model.forward_v1_features(x)
        logits_from_v1 = model.forward_from_v1_features(v1)
        logits_from_override = model.forward_with_v1_override(x, v1)

    assert v1.shape == (1, 512, 56, 56)
    assert logits_from_v1.shape == (1, 1000)
    assert logits_from_override.shape == (1, 1000)


def test_v1_convnext_noise_train_only_behavior():
    torch.manual_seed(0)
    model = V1ConvNeXt(
        backbone_name="convnext_small",
        input_size=224,
        noise_mode="gaussian",
        v1_noise_train_only=True,
    )
    x = torch.randn(1, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        eval_default = model.forward_v1_features(x)
        eval_clean = model.forward_v1_features(x, apply_noise=False)
    assert torch.allclose(eval_default, eval_clean)

    model.train()
    with torch.no_grad():
        train_default = model.forward_v1_features(x)
        train_clean = model.forward_v1_features(x, apply_noise=False)
    assert not torch.allclose(train_default, train_clean)
