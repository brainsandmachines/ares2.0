# MADRY PGD-3 ConvNeXt Timing

One epoch ImageNet adversarial training with `attack_criterion=madry`, `attack_it=3`, `attack_norm=linf`.

| model | time for 1 epoch | batch size | GPU |
|---|---:|---:|---|
| `convnext_small` | 2:19:19 | 256 | NVIDIA RTX 6000 Ada Generation |
| `convnext_base` | 3:22:39 | 256 | NVIDIA RTX 6000 Ada Generation |
| `convnext_large` | 6:23:59 | 192 | NVIDIA RTX 6000 Ada Generation |

