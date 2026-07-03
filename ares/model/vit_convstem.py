"""ViT + ConvStem models, ported from nmndeep/revisiting-at (utils_architecture.py).

"Revisiting Adversarial Training for ImageNet" (Singh, Croce, Hein; NeurIPS 2023):
the patchify stem (single stride-16 conv in patch_embed.proj) is replaced by a
stack of stride-2 3x3 convs (total stride 16), which improves adversarial
robustness and generalization to unseen threat models. The ViT itself is stock
timm; only patch_embed.proj is swapped, so token count (14x14) is unchanged.

ViT-M follows the paper and uses timm's DeiT3-medium (embed_dim 512, depth 12,
heads 8) as the base for the ConvStem variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from timm.models.registry import register_model


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvBlock(nn.Module):
    """ConvStem for embed_dim = siz * end_siz (384 for ViT-S). Total stride 16."""

    expansion = 1

    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock, self).__init__()
        self.planes = siz
        fin_dim = self.planes * end_siz if fin_dim != 432 else 432
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes, self.planes * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 2, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes * 2, self.planes * 4, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 4, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes * 4, self.planes * 8, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 8, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes * 8, fin_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.stem(x)


class ConvBlock2(nn.Module):
    """ConvStem with fixed 512 output dim, for ViT-M (DeiT3-medium). Total stride 16."""

    expansion = 1

    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock2, self).__init__()
        self.planes = siz
        fin_dim = self.planes * end_siz if fin_dim != 432 else 432
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes, self.planes * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 2, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes * 2, self.planes * 4, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 4, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes * 4, self.planes * 8, kernel_size=3, stride=2, padding=1),
            LayerNorm(self.planes * 8, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(self.planes * 8, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.stem(x)


@register_model
def vit_s_cvst(pretrained=False, **kwargs):
    """ViT-S/16 with ConvStem (revisiting-at 'vit_s', not_original=True)."""
    model = create_model('vit_small_patch16_224', pretrained=pretrained, **kwargs)
    model.patch_embed.proj = ConvBlock(48, end_siz=8)
    return model


@register_model
def vit_m_cvst(pretrained=False, **kwargs):
    """ViT-M (DeiT3-medium) with ConvStem (revisiting-at 'vit_m', not_original=True)."""
    model = create_model('deit3_medium_patch16_224', pretrained=pretrained, **kwargs)
    model.patch_embed.proj = ConvBlock2(48)
    return model
