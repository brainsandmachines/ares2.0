# `ares` Runtime Optimization Contract

This skill treats the current repo behavior as the protocol to preserve.

## Attack Contract

Source files:

- `robust_training/configs/attacks/adv.yaml`
- `robust_training/adversarial_training.py`
- `ares/utils/adv.py`
- `ares/utils/train_loop.py`

Protected behaviors:

- `attack_it=3`
- `v1_attack_it=3`
- `Linf` pixel-space default step derives as `attack_eps / attack_it`
- `L2` pixel-space default step derives as `2 * attack_eps / attack_it`
- `L1` default step path stays `l2_norm`
- `LinfStep.step()` uses `sign(grad) * step_size`
- `L2Step.step()` uses per-sample normalized `L2` gradient direction
- `L1Step.step()` protected path is `l2_norm` only
- `Linf`, `L2`, and `L1` projection behaviors must not change
- `TRADES` keeps random start enabled for non-`L1` attacks

## Dataset Contract

Source files:

- `robust_training/configs/dataset/imagenet.yaml`
- `ares/utils/dataset.py`
- `robust_training/adversarial_training.py`

Protected behaviors:

- augmentation arguments must stay identical unless user opts in
- mixup arguments must stay identical unless user opts in
- `mixup_off_epoch` rebuild behavior must stay intact
- benchmark dataset should prefer `imagenet_sample` when available

Current ImageNet dataset settings that must not drift silently:

- `aa=rand-m9-mstd0.5-inc1`
- `reprob=0`
- `mixup_active=true`
- `mixup=0.8`
- `cutmix=1.0`
- `mixup_prob=0.5`
- `mixup_switch_prob=0.5`
- `mixup_mode=elem`
- `mixup_off_epoch=175`

## GradNorm Contract

Source files:

- `robust_training/configs/attacks/adv.yaml`
- `robust_training/adversarial_training.py`
- `ares/utils/train_loop.py`
- `ares/utils/gradnorm.py`

Protected behaviors:

- GradNorm remains a regularization path, not adversarial training
- `gradnorm_penalty_norm=l2` must be explicitly covered by tests and benchmarks
- candidate runtime work may reduce overhead but must not change the regularization meaning

## Pixel-Space Protocol Coverage For This Skill

The skill must support local and Slurm benchmarking for these pixel-space protocol families:

- ConvNeXt Small `linf` Madry
- ConvNeXt Small `l2` Madry
- ConvNeXt Small `l1` Madry
- ConvNeXt Small `linf` TRADES
- ConvNeXt Small `l2` TRADES
- ConvNeXt Small `l1` TRADES

For TRADES, the main repo contract stays unchanged:

- current training flow uses random start when `attack_norm != 'l1'`
- current training flow disables random start for `l1` TRADES

The skill may benchmark additive TRADES random-start variants, but only through additive wrappers in the benchmark runner. That does not change the main repo contract.

## V1 Feature-Attack Contract

Source files:

- `robust_training/configs/model/convnext_small_v1.yaml`
- `robust_training/adversarial_training.py`
- `ares/utils/adv.py`

Protected behaviors:

- `convnext_small_v1` uses feature-space attack math when `attack_domain=v1_feature`
- V1 noise must stay rejected for adversarial training
- this skill protects `attack_norm=l2` for both Madry and TRADES V1 feature adversarial training
