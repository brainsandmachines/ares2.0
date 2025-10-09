This folder contains Hydra configuration groups for training.

Layout:
- `config.yaml` - main composing config (defaults list).
- `training/convnext_small.yaml` - schedule, augmentation, and misc training options.
- `model/convnext_small.yaml` - model-specific settings.
- `dataset/imagenet.yaml` - dataset paths and normalization.
- `optimizer/adamw.yaml` - optimizer defaults.
- `attacks/adv.yaml` - adversarial training / attack settings.

How to run:
Use the Hydra entrypoint `robust_training/hydra_runner.py` which converts the composed
Hydra config into the argparse.Namespace expected by the existing training code.

Examples:
  python robust_training/hydra_runner.py
  python robust_training/hydra_runner.py training.epochs=200 optimizer.weight_decay=0.1

Notes:
- Field names follow the original `convnext_small.yaml` layout. Override any field via the CLI.
- If you want to add more models/datasets/optimizers/attacks, add files under the respective group folders.
