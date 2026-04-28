import torch
from torch import nn
from timm.models import create_model

from ares.model.v1_block import build_v1_block


class V1ConvNeXt(nn.Module):
    """ConvNeXt with a fixed V1 front-end inserted ahead of the backbone trunk."""

    def __init__(
        self,
        backbone_name="convnext_small",
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
        global_pool=None,
        bn_momentum=None,
        bn_eps=None,
        input_size=224,
        v1_noise_train_only=True,
        **v1_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.v1_noise_train_only = v1_noise_train_only
        self.v1_block = build_v1_block(image_size=input_size, **v1_kwargs)

        backbone = create_model(
            backbone_name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=drop_block_rate,
            global_pool=global_pool,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )
        self.backbone = backbone

        stem = getattr(backbone, "stem", None)
        if stem is None or len(stem) < 2:
            raise ValueError(f"Backbone {backbone_name} does not expose the expected ConvNeXt stem")

        stem_conv = stem[0]
        stem_norm = stem[1]
        stem_channels = stem_conv.out_channels

        self.feature_adapter = nn.Conv2d(self.v1_block.out_channels, stem_channels, kernel_size=1, bias=False)
        self.backbone.stem = nn.Sequential(stem_norm)

    @property
    def num_classes(self):
        return self.backbone.num_classes

    def _should_apply_v1_noise(self, apply_noise=None):
        if apply_noise is not None:
            return bool(apply_noise)
        if not self.v1_noise_train_only:
            return True
        return self.training

    def forward_v1_features(self, x, apply_noise=None):
        return self.v1_block(x, add_noise=self._should_apply_v1_noise(apply_noise))

    def forward_from_v1_features(self, v1_features):
        x = self.feature_adapter(v1_features)
        return self.backbone.forward_head(self.backbone.forward_features(x))

    def forward_with_v1_override(self, x, v1_features_override):
        return self.forward_from_v1_features(v1_features_override)

    def forward_features(self, x):
        x = self.forward_v1_features(x)
        x = self.feature_adapter(x)
        return self.backbone.forward_features(x)

    def forward_head(self, x, pre_logits=False):
        return self.backbone.forward_head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        return self.forward_head(x)
