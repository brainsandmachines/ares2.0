# Madry ResNet50 PGD Eval

This folder contains an isolated PGD evaluation flow for original Madry ResNet50 checkpoints under:

- `/storage/test/bml_group/tomerash/madry_orig_robustmodels`

Only files matching `resnet50_l2_eps*.ckpt` are selected. `wide_resnet*` checkpoints are excluded.

## What it runs

- PGD norms: `linf,l2,l1`
- Epsilons: `0,0.01,0.03,0.05,0.1,0.25,0.5,1,3,5`
- Auto input-mode detection (`--input-mode auto`) to decide whether the checkpoint behaves best with:
  - normalized ImageNet input, or
  - raw `[0,1]` input

The detector probes a few batches in both modes and selects the better clean top-1 mode (with checkpoint normalizer-key tie-break).

Array launcher pre-check behavior:

- Before dispatching each model, `run_single_model_eval.sh` scans the checkpoint for a `normalizer` signature.
- If found, it forces `--input-mode raw` immediately (no normalization probe).

## Single model run

```bash
python resnet_pgd_exp/madry_resnet50_pgd_eval.py \
  --checkpoint /storage/test/bml_group/tomerash/madry_orig_robustmodels/resnet50_l2_eps0.1.ckpt \
  --val-dir /storage/test/bml_group/tomerash/datasets/imagenet/val \
  --out-dir /storage/test/bml_group/tomerash/madry_orig_robustmodels/pgd_eval_resnet50/resnet50_l2_eps0.1 \
  --device cuda \
  --input-mode auto
```

## Slurm array run

```bash
sbatch sbatches/resnet_pgd_eval_madry.sbatch
```

Optional overrides:

```bash
VAL_DIR=/storage/test/bml_group/tomerash/datasets/imagenet/val \
MODELS_ROOT=/storage/test/bml_group/tomerash/madry_orig_robustmodels \
OUT_ROOT=/storage/test/bml_group/tomerash/madry_orig_robustmodels/pgd_eval_resnet50 \
LOCAL_OUT_ROOT=/home/ashtomer/projects/ares/resnet_pgd_exp/results \
INPUT_MODE=auto \
sbatch sbatches/resnet_pgd_eval_madry.sbatch
```

## Outputs

For each model:

- `<out_dir>/pgd_validation_results.csv`
- `<out_dir>/madry_resnet50_pgd_eval-YYYY-MM-DD.log`

Additionally, outputs are mirrored per-model into:

- `/home/ashtomer/projects/ares/resnet_pgd_exp/results/<model_stem>/`

CSV includes detection fields:

- `input_mode_detected`
- `normalizer_in_ckpt`
