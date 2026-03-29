# V1 Metric Pipeline (Isolated Research Module)

This folder contains a fully additive, self-contained experimental pipeline for:
1. deterministic V1-inspired feature extraction,
2. V1-based perceptual distance,
3. practical adversarial generation using `CE - lambda * d_V1`,
4. a minimal adversarial-training scaffold.

All code is isolated under `research/v1_metric_pipeline/` and does not modify existing training/attack/model code paths.

## What Was Implemented

- `models/v1_feature_extractor.py`
  - Fixed Gabor filter bank.
  - Simple-cell nonlinearities (phase-sensitive rectification).
  - Complex-cell nonlinearities from quadrature phase pairs (`0`, `pi/2`).
  - Optional pooling.
  - Optional per-channel normalization via fixed stats.

- `metrics/v1_metric.py`
  - Differentiable batch metric:
    - `d_V1 = alpha * normalized_L2(simple) + beta * normalized_L2(complex)`
  - Defaults: `beta >= alpha` (`alpha=0.3`, `beta=0.7`).

- `attacks/v1_penalty_attack.py`
  - Practical iterative attack maximizing:
    - `CE(model(x_adv), y) - lambda_v1 * d_V1(x, x_adv)`
  - Constraints: `[0,1]` box clamp + optional per-pixel guardrail around original image.
  - Optional cheap shrink-back mode to enforce a V1 budget without expensive exact projection.

- `scripts/train_with_v1_metric.py`
  - Minimal, isolated adversarial-training scaffold (uses local `TinyCNN`, no production imports).

- Utilities
  - Gabor visualization.
  - V1 response-map visualization.
  - Hyperparameter printout.
  - Demo script that computes metric and generates an adversarial sample.

## VOne-Inspired Assumptions Used

- Orientation bank over `[0, pi)` with default 8 orientations.
- Quadrature phases `[0, pi/2]`.
- Spatial frequency sampling in cycles-per-degree with conversion to pixel-space using configurable field-of-view (default `8 deg @ 224 px`).
- Deterministic metric representation (no stochastic VOne noise).

## Simplifications vs Full VOneNet

- No VOne stochasticity in the metric/attack objective.
- No full VOneNet backbone replacement or ImageNet training pipeline reproduction.
- No exact nonlinear projection onto a perceptual ball; uses practical penalty + optional shrink-back.

## Quick Start

Print resolved config:
```bash
python -m research.v1_metric_pipeline.scripts.print_hparams \
  --config research/v1_metric_pipeline/configs/v1_defaults.yaml
```

Visualize Gabor kernels:
```bash
python -m research.v1_metric_pipeline.scripts.visualize_gabors \
  --config research/v1_metric_pipeline/configs/v1_defaults.yaml \
  --out research/v1_metric_pipeline/outputs/gabor_grid.png
```

Run full demo (two images required):
```bash
python -m research.v1_metric_pipeline.scripts.demo_v1_pipeline \
  --config research/v1_metric_pipeline/configs/demo.yaml \
  --image-a /path/to/image_a.png \
  --image-b /path/to/image_b.png \
  --output-dir research/v1_metric_pipeline/outputs/demo_run
```

Train scaffold (research-only):
```bash
python -m research.v1_metric_pipeline.scripts.train_with_v1_metric \
  --config research/v1_metric_pipeline/configs/train_scaffold.yaml
```

## Testing

Run isolated tests only:
```bash
pytest -q research/v1_metric_pipeline/tests
```
