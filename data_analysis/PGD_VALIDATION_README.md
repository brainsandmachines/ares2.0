PGD Validation Sweep

Editable paths for cluster runs:
- In `data_analysis/run_pgd_validation_sweep.py`:
  - `DEFAULT_MODELS_DIR`
  - `DEFAULT_VAL_DIR`
  - `DEFAULT_OUT_CSV`
  - `DEFAULT_LOG_PATH`
- In `data_analysis/plot_pgd_validation.py`:
  - `DEFAULT_CSV`
  - `DEFAULT_OUT_DIR`

Requested epsilon setup:
- `EPS_VALUES = [0.1, 0.5, 1, 2, 4, 8, 16]`
- For Linf only: `epsilon_eval = epsilon_input / 255.2`

Run examples:
```bash
conda run -n contstim python data_analysis/run_pgd_validation_sweep.py \
  --models-dir /mnt/data/robustness_models/for_experiment \
  --val-dir /mnt/data/datasets/imagenet/val \
  --out-csv data_analysis/pgd_validation_results.csv \
  --log-path data_analysis/pgd_validation.log \
  --device cuda
```

Quick sanity run (small):
```bash
conda run -n contstim python data_analysis/run_pgd_validation_sweep.py \
  --device cpu --max-models 1 --max-batches 1 --batch-size 4
```

Plot (combined + per-norm):
```bash
conda run -n contstim python data_analysis/plot_pgd_validation.py \
  --csv data_analysis/pgd_validation_results.csv \
  --out-dir data_analysis/plots \
  --make-norm-plots
```

Plot folder structure:
- Combined: `data_analysis/plots/<category>/combined/init<init>/<model>_combined.png`
- Single norm: `data_analysis/plots/<category>/<norm>/init<init>/<model>_<norm>.png`
