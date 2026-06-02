This folder contains Hydra configuration groups for training.

Layout:
- `config.yaml` - main composing config (defaults list).
- `training/convnext_small.yaml` - schedule, augmentation, and misc training options.
- `model/convnext_small.yaml` - model-specific settings.
- `dataset/imagenet.yaml` - dataset paths and normalization.
- `optimizer/adamw.yaml` - optimizer defaults.
- `attacks/adv.yaml` - adversarial training / attack settings.

How to run:
Use the direct Hydra entrypoint:

Examples:
  python -m robust_training.adversarial_training
  python -m robust_training.adversarial_training training.epochs=200 optimizer.weight_decay=0.1

Notes:
- Override grouped Hydra fields via the CLI, for example `training.epochs=200`.
- If you want to add more models/datasets/optimizers/attacks, add files under the respective group folders.
