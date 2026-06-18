# RIG vs ARES GradNorm Training Comparison

This note compares the public RIG ImageNet `GradNorm - ResNet50+GeLU` model against the ARES GradNorm runs:

- `convnext_small_gradnorm_l1_1_init1`
- `convnext_small_gradnorm_l2_1_init1`
- `convnext_small_gradnorm_l2_16_init1`

The main conclusion is that the successful RIG model is not just "the same GradNorm objective on a different model." It differs in architecture, initialization, exact DBP penalty, penalty scaling, cap behavior, augmentation, precision, gradient clipping, and schedule. The largest implementation-level mismatch is that RIG trains with `0.8 * CE + 1.2 * alpha * DBP_L1`, while the current ARES GradNorm loop trains with `CE + capped(alpha * DBP)` and does not apply the saved `ce_weight` or `gradnorm_weight` config values.

## Sources

RIG public sources:

- Repo: <https://github.com/adrianrm99/robustness_input_gradients>
- RIG table result: README reports `GradNorm - ResNet50+GeLU` at 60.34 clean and 30.00 AutoAttack Linf 4/255.
- Training config: `configs_train/gradnorm_resnet_gelu.yaml`
- Finetune config: `configs_train/finetune_resnet_gelu.yaml`
- Loss implementation: `input_norm_losses.py`
- Training loop: `adversarial_training.py`

Local ARES sources:

- `ares/utils/gradnorm.py`
- `ares/utils/train_loop.py`
- `robust_training/adversarial_training.py`
- `robust_training/configs/attacks/adv.yaml`
- Saved run configs under `/groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models/`
- RIG evaluation outputs under `research/gradnorm/results/rig_gradnorm_resnet50_gelu/`

## Executive Takeaways

1. RIG uses ResNet50 with ReLU replaced by GELU, not ConvNeXt-small. It also starts from a 4-epoch ImageNet-pretrained ResNet50+GELU finetune checkpoint before GradNorm training.
2. RIG's released GradNorm model uses only an L1 input-gradient DBP penalty. Your L2 experiments use a squared-L2 DBP penalty, so they are not a direct variant of RIG's objective.
3. RIG's DBP defaults to `eps=4/255`. Your `l1_1` and `l2_1` runs use `eps=1/255`; `l2_16` uses `16/255`.
4. RIG applies explicit objective weights: `0.8 * CE + 1.2 * alpha * DBP`. The current ARES GradNorm branch does not apply `ce_weight=0.8` or `gradnorm_weight=1.2`.
5. ARES caps the regularizer to `gradnorm_max_reg_to_ce_ratio * CE`, with your saved runs using `1.0`. RIG does not have this cap in the public training loop.
6. RIG disables AMP and clips gradients at norm `1.0`. Your runs use native AMP and no gradient clipping.
7. RIG uses stronger regularization-oriented augmentation: `color_jitter=0.4`, random erasing `0.25`, `mixup_prob=1.0`, and batch mixup. Your runs use `color_jitter=0`, no random erasing, `mixup_prob=0.5`, element mixup, and turn mixup off only at epoch 175.

## Side-by-Side Recipe

| Dimension | RIG GradNorm ResNet50+GELU | ARES `l1_1` | ARES `l2_1` | ARES `l2_16` |
|---|---:|---:|---:|---:|
| Model | `resnet50` | `convnext_small` | `convnext_small` | `convnext_small` |
| Activation change | Replace ReLU with GELU | ConvNeXt native activations | ConvNeXt native activations | ConvNeXt native activations |
| ImageNet classes | 1000 | 1000 | 1000 | 1000 |
| Initialization | 4-epoch finetuned ResNet50+GELU checkpoint | Saved run config, no RIG-style GELU finetune stage | Same | Same |
| GradNorm epochs | 83 | Intended 200, stopped after epoch 8 | 200 | 200 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Weight decay | 0.05 | 0.05 | 0.05 | 0.05 |
| Gradient clipping | `clip_grad=1.0`, norm | `null` | `null` | `null` |
| Native AMP | False | True | True | True |
| Per-GPU batch | 64 | Run stores batch 384 | Run stores batch 384 | Run stores batch 384 |
| Global batch used for LR | 256 from 4 GPUs x 64 | 384 | 384 | 384 |
| LR base `lrb` | `1.25e-3` | `1.0e-3` | `1.0e-3` | `1.0e-3` |
| Effective LR | `0.000625` | `0.00075` | `0.00075` | `0.00075` |
| Scheduler | Cosine | Cosine | Cosine | Cosine |
| Warmup | 5 epochs | 20 epochs | 20 epochs | 20 epochs |
| Min LR | `1e-5` | `1e-5` | `1e-5` | `1e-5` |
| Model EMA | True, decay `0.9998` | True, decay `0.9998` | True, decay `0.9998` | True, decay `0.9998` |
| Loss function | DBP | DBP | DBP | DBP |
| DBP norm | L1 only | L1 | Squared L2 | Squared L2 |
| DBP epsilon | Default `4/255` | `1/255` | `1/255` | `16/255` |
| CE weight | Used: `0.8` | Config has `0.8`, not used by current loop | Config has `0.8`, not used by current loop | Config has `0.8`, not used by current loop |
| GradNorm weight | Used: `1.2` | Config has `1.2`, not used by current loop | Config has `1.2`, not used by current loop | Config has `1.2`, not used by current loop |
| Alpha schedule | `[0.0, 0.2, 1.0]`, reaches 1 after about 5 epochs | Starts `0.1`, ramps to 1 over 9 epochs | Same | Same |
| Reg-to-CE cap | None found | `gradnorm_max_reg_to_ce_ratio=1.0` | `1.0` | `1.0` |
| Color jitter | `0.4` | `0` | `0` | `0` |
| Random erasing | `0.25` | `0` | `0` | `0` |
| Mixup / CutMix | `0.8` / `1.0` | `0.8` / `1.0` | `0.8` / `1.0` | `0.8` / `1.0` |
| Mixup probability | `1.0` | `0.5` | `0.5` | `0.5` |
| Mixup mode | `batch` | `elem` | `elem` | `elem` |
| Mixup off epoch | `0` | `175` | `175` | `175` |
| Label smoothing | `0.1` | `0.1` | `0.1` | `0.1` |
| AutoAugment | `rand-m9-mstd0.5-inc1` | Same | Same | Same |

## Objective-Level Differences

RIG DBP implementation:

```text
DBP(gradients, inputs) = (eps / std) * batch_size * mean(sum(abs(gradients)))
default eps = 4/255
std = 0.225
```

RIG GradNorm training branch:

```text
loss = ce_weight * CE
gradient = d(CE) / d(input)
loss_reg = DBP(gradient, input)
alpha = clamp(alpha0 + progress * alpha_slope, max=alpha_max)
loss += gradnorm_weight * alpha * loss_reg
```

For `gradnorm_resnet_gelu.yaml`, this is:

```text
loss = 0.8 * CE + 1.2 * alpha * DBP_L1
alpha = min(0.0 + progress_epochs * 0.2, 1.0)
```

Current ARES DBP implementation:

```text
if penalty_norm == "l1":
    penalty = sum(abs(gradients))
elif penalty_norm == "l2":
    penalty = sum(gradients ** 2)
DBP = (eps / std) * batch_size * mean(penalty)
```

Current ARES GradNorm training branch:

```text
ce_loss = loss_fn(output, target)
gradient = d(ce_loss) / d(input)
alpha = compute_gradnorm_alpha(...)
raw_loss_reg = DBP(gradient, input) * alpha
loss_reg = min(raw_loss_reg, gradnorm_max_reg_to_ce_ratio * ce_loss)
loss = ce_loss + loss_reg
```

This is not equivalent to the RIG objective. In particular:

- `ce_weight` is present in your saved configs as `0.8`, but the current ARES GradNorm branch does not multiply CE by it.
- `gradnorm_weight` is present in your saved configs as `1.2`, but the current ARES GradNorm branch does not multiply the DBP term by it.
- `gradnorm_max_reg_to_ce_ratio=1.0` caps the effective regularization. If raw DBP is larger than CE, the local objective silently scales DBP down.
- The local L2 DBP is squared L2, not L2 norm. It penalizes large gradient coordinates differently from RIG's L1 DBP and changes the scale substantially.

## Outcome Comparison

RIG public result:

| Model | Clean | AutoAttack Linf 4/255 |
|---|---:|---:|
| RIG GradNorm ResNet50+GELU | 60.34 | 30.00 |

Local RIG evaluation in `research/gradnorm`:

| Eval | Images | Clean | Notes |
|---|---:|---:|---|
| Full ImageNet validation | 50,000 | 59.74 top-1 / 82.398 top-5 | `clean_full_val_results.json` |
| AutoAttack selected subset | 256 | 57.8125 clean | Same 16 batches of 16 images |

Local RIG AutoAttack subset results:

| Norm | Epsilon | Robust accuracy |
|---|---:|---:|
| Linf | 0.1/255 | 57.42 |
| Linf | 0.5/255 | 54.30 |
| Linf | 1/255 | 50.78 |
| Linf | 2/255 | 44.92 |
| Linf | 4/255 | 30.08 |
| Linf | 6/255 | 15.23 |
| Linf | 8/255 | 7.42 |
| L2 | 0.1 | 55.86 |
| L2 | 0.5 | 52.73 |
| L2 | 1 | 48.05 |
| L2 | 2 | 33.59 |
| L2 | 4 | 12.11 |
| L2 | 6 | 3.12 |
| L2 | 8 | 0.39 |

Your ARES GradNorm outcomes:

| Run | Training outcome | Clean / validation behavior | AutoAttack behavior |
|---|---|---|---|
| `convnext_small_gradnorm_l1_1_init1` | Stopped after epoch 8 | Last eval top-1 about 0.222 | No useful robustness result; run effectively crashed/collapsed |
| `convnext_small_gradnorm_l2_1_init1` | Completed 200 epochs | Final eval top-1 about 79.34; AA CSV clean 81.45 on 1024 images | Linf 1/255: 0.195; Linf >=2/255: 0; L2 1: 0.098; L2 >=2: 0 |
| `convnext_small_gradnorm_l2_16_init1` | Completed 200 epochs | Final eval top-1 about 79.51; AA CSV clean 83.01 on 1024 images | Linf 1/255: 0.781; Linf >=2/255: 0; L2 1: 0.098; L2 >=2: 0 |

The L2 runs are high-clean-accuracy models, but their AutoAttack scores show that the squared-L2 DBP objective as currently implemented did not produce the RIG-style robust behavior.

## Most Important Differences To Test Next

The next experiments should separate implementation mismatch from scientific choices. The first goal should be a faithful RIG reproduction inside ARES before trying ConvNeXt variants.

Recommended order:

1. Add or configure a faithful RIG-style objective:
   - `loss = 0.8 * CE + 1.2 * alpha * DBP_L1`
   - DBP epsilon `4/255`
   - alpha schedule `[0.0, 0.2, 1.0]`
   - no reg-to-CE cap, or set cap high enough that it does not bind
2. Reproduce architecture and initialization:
   - ResNet50
   - replace ReLU with GELU
   - start from the 4-epoch ResNet50+GELU finetune checkpoint, not from the current ConvNeXt recipe
3. Match stabilizers:
   - disable native AMP for the first reproduction
   - use `clip_grad=1.0`
   - use RIG's 5 warmup epochs, 83 GradNorm epochs, and global batch 256 if resources allow
4. Match augmentation:
   - `color_jitter=0.4`
   - `reprob=0.25`
   - `mixup_prob=1.0`
   - `mixup_mode=batch`
   - `mixup_off_epoch=0`
5. Only after the ResNet50+GELU reproduction works, move one factor at a time:
   - ConvNeXt-small with faithful RIG objective
   - ConvNeXt-small with/without cap
   - L1 DBP versus squared-L2 DBP
   - AMP on/off
   - gradient clipping on/off

## Interpretation

The current evidence does not show that input-gradient regularization failed in general. It shows that your local ConvNeXt GradNorm recipe differs from RIG in several high-impact ways, and that the local L2 DBP variant is not a drop-in replacement for the RIG L1 DBP objective.

The strongest hypothesis is that the RIG robustness comes from the combination of ResNet50+GELU initialization, L1 DBP at `4/255`, explicit `0.8/1.2` loss weights, no CE-ratio cap, gradient clipping, no AMP, and the RIG augmentation schedule. The current ARES loop should be made objective-equivalent to RIG before drawing conclusions about architecture or L1 versus L2 penalties.
