# AGENTS.md

## Project Scope
- Repository: `ares` (adversarial robustness for classification, detection, and robust training).
- Main working areas:
  - `robust_training/` for training and Hydra configs.
  - `data_analysis/` for PGD/AutoAttack/final-eval scripts.
  - `sbatches/` and `sh_files/` for Slurm job orchestration.
  - `classification/` and `detection/` are present in the repo but out of scope for this workflow.

## Environment Awareness (Important)
- This repo may run in either:
  - A Slurm cluster environment.
  - Botero Linux environment.
- Do not assume one fixed absolute path.
- Before running jobs, adapt all dataset and project paths to the current machine.
- Prefer configurable CLI/Hydra overrides over hardcoded edits when possible.

## Path Rules For Agents
- If `SLURM_JOB_ID`/`SLURM_PROCID` is present, assume a Slurm execution context.
- Always verify and update these path categories before launching:
  - Dataset roots (`train_dir`, `eval_dir`, `val_dir`).
  - Checkpoint/model roots.
  - Output/log directories.
  - Repo absolute paths (`/home/.../projects/ares`), if scripts rely on them.
- Keep project-local relative paths when possible; use absolute paths only when required by cluster job scripts.

## Dataset Paths Found In This Repo

### ImageNet (training/eval)
- `/storage/test/bml_group/tomerash/datasets/imagenet/train/`
- `/storage/test/bml_group/tomerash/datasets/imagenet/val/`
- `~/datasets/imagenet/train`
- `~/datasets/imagenet/val`
- `/mnt/data/datasets/imagenet_sample/train`
- `/mnt/data/datasets/imagenet_sample/val`
- `/mnt/data/datasets/imagenet/val`

## Files Where Dataset Paths Are Defined
- `robust_training/configs/dataset/imagenet.yaml`
- `robust_training/train_configs/*.yaml`
- `sbatches/botero_tests.sbatch`
- `data_analysis/run_pgd_validation_sweep.py`
- `data_analysis/autoattack_eval.py`
- `data_analysis/PGD_VALIDATION_README.md`
- `data_analysis/FINAL_EVAL_README.md`

## Recommended Agent Workflow Before Running Jobs
1. Detect environment (Slurm vs Botero/local Linux).
2. Confirm active dataset roots exist on this machine.
3. Override dataset paths in command line/Hydra (`dataset.train_dir=...`, `dataset.eval_dir=...`, `--val-dir ...`) instead of editing many files.
4. Confirm output/checkpoint directories are writable.
5. Run.

## Notes
- Many scripts in `sbatches/`, `sh_files/`, and `job_manager/` contain user-specific absolute paths under `/home/ashtomer/...`; adapt them per machine.
- Preserve existing behavior unless the task explicitly asks for path refactoring.
