# AGENTS.md

This file provides guidance to any coding agent (Claude Code, Codex, etc.) working in this
repository. It is the single source of truth for project context and environment rules.

## What this repo is

`ares` is a fork of [thu-ml/ares2.0](https://github.com/thu-ml/ares2.0) (adversarial robustness
benchmarking for image classification/detection). The original library (`ares/attack`, `ares/model`,
`ares/dataset`, `ares/defense`, `classification/`, `detection/`) is upstream code — treat it as a
stable dependency, not the focus of day-to-day work.

The actual active project built on top of it is an **adversarial-training research + orchestration
system**: a Hydra training entrypoint (`robust_training/`), evaluation/plotting scripts
(`data_analysis/`), and two independent CSV-driven job-queue managers that keep GPU clusters busy
running training campaigns (`slurm_job_manager/`, `aircc/aircc_job_manager/`).

## Three systems — always know which one you're targeting

1. **Botero** — this machine. Where all editing, git, and orchestration happens. Never edit code
   through `~/slurm_mount` or `~/aircc_mount` (both are read-only sshfs mounts for inspecting
   remote logs/checkpoints only).
2. **Slurm cluster** (`ssh slurm`, remote repo `/home/ashtomer/projects/ares`) — run via
   `slurm_job_manager/`. Partitions `rtx_pro_6000` (96GB) / `rtx6000`. One array task = one GPU =
   one model, single-process train→final_eval→plot.
3. **AIRCC B200 allocation** (`sandbox`) — run via `aircc/aircc_job_manager/`. 16 GPUs, 2 training
   lifecycles per GPU (memory-fraction capped).

Both job managers share the same philosophy: **the CSV is ground truth** (one column per Hydra
override), the SQLite DB holds only operational state (claim status, progress, best checkpoint).
`slurm_job_manager`'s DB schema is a superset of AIRCC's so `aircc_job_manager/progress.py`'s
in-training hooks work unchanged against either (the only cross-package import between them).

`archive/legacy_job_manager_2026-06-19/` and `archive/orchestrator_2026-07-10/` are retired
predecessors — left for reference, not in use; don't resurrect or extend them.

## Commands

**Run adversarial training directly (Hydra entrypoint):**
```bash
python -m robust_training.adversarial_training model=convnext_small training.epochs=200 \
    attacks.attack_norm=linf attacks.attack_eps=8
```
Override any grouped config field from the CLI (`training.*`, `model.*`, `dataset.*`, `optimizer.*`,
`attacks.*`, `epsilon_schedule.*`, `continuation.*`, `checkpointing.*`). Config groups live under
`robust_training/configs/` (see `robust_training/configs/README.md`). `final_eval=True` (default)
runs AutoAttack over best/last/advbest checkpoints and writes a comparison plot after training, in
the same process.

**Tests:**
```bash
python -m pytest tests/ -q                    # core library / training-loop tests
python -m pytest slurm_job_manager/tests/ -q  # job-manager DB/lifecycle tests
python -m pytest tests/test_gradnorm.py -q    # single file
python -m pytest tests/test_gradnorm.py::test_name -q  # single test
```
No `pytest.ini`/`pyproject.toml` — plain pytest discovery. A couple of `tests/test_*.sh` files
exercise shell launcher logic directly (not via pytest).

**Job managers (read-only status / dry runs are safe to run anytime):**
```bash
python -m slurm_job_manager.status --db slurm_job_manager/jobs.sqlite
python -m slurm_job_manager.controller --db slurm_job_manager/jobs.sqlite --dry-run
python -m aircc.aircc_job_manager.status
```
Seeding/submitting new campaign rows or sbatch submissions are not side-effect-free — confirm with
the user before running `seed.py`/`seed_db.py` writes or `sbatch`.

## Architecture

- **`ares/`** — upstream library: `attack/` (attack implementations incl. `attack/autoattack/`),
  `model/` (architectures + `imagenet_model_zoo.py`/`cifar_model_zoo.py`), `dataset/`, `defense/`.
  `ares/utils/` is where this project's own extensions to the core library live (not upstream):
  `train_loop.py` (the actual epoch loop), `continuation.py` (checkpoint-resume + epsilon-carry
  logic for continuing a run at a new eps/norm), `epsilon_schedule.py` (linear eps ramps),
  `gradnorm.py` (GradNorm multi-task loss balancing), `dvd.py` (developmental visual diet /
  age-curve data augmentation), `final_eval_helpers.py` (post-training AutoAttack + completion
  detection so reruns don't redo finished evals), `runtime_probe.py` (perf timing).
- **`robust_training/adversarial_training.py`** — the single Hydra entrypoint (`main()` /
  `hydra_main()`) that every job manager and sbatch script ultimately calls. Wires together dataset,
  model, optimizer/scheduler, distributed init, the epsilon schedule, continuation-checkpoint
  loading, the train loop, validation, final eval, and (via `aircc/aircc_job_manager/progress.py`)
  writes progress/heartbeat back to whichever job DB launched it — a no-op unless `AIRCC_DB`/
  `AIRCC_MODEL_ID` env vars are set.
- **`robust_training/configs/`** — Hydra config groups (`model/`, `training/`, `dataset/`,
  `optimizer/`, `attacks/`, `epsilon_schedule/`, `continuation/`, `checkpointing/`, `machine/`,
  `lr_scheduler/`, `dist/`); `config.yaml` is the composing default.
- **`slurm_job_manager/`** and **`aircc/aircc_job_manager/`** — parallel but independent CSV+SQLite
  job queues (see each package's own README for its file-by-file breakdown: `db.py` atomic claim,
  `lifecycle.py` command-building + subprocess run, `controller.py`/`job_manager.py` main loop,
  `csv_spec.py` column schema, `status.py` dashboard, `seed.py`/`seed_db.py` to add jobs). Don't
  assume behavior is shared between the two beyond the DB schema/progress-hook overlap noted above.
- **`data_analysis/`** — standalone eval/plotting scripts (`final_eval.py`, `autoattack_eval.py` and
  variants, `training_plots.py`, shape-bias analysis) generally run against already-trained
  checkpoints, independent of the job managers.
- **`sbatches/`** (Slurm cluster paths) vs **`sbatches_botero/`** (Botero-local sbatch variants) —
  mirror the same jobs for the two clusters; check which one a script belongs to before editing.
- **`.agents/skills/`** — repo-local agent skills (`submit-training-jobs`, `training-runtime-optimizer`)
  with their own protocol-preservation constraints (e.g. never silently change attack step/projection
  semantics, augmentation, mixup, or batch size as an "optimization").

## Conventions worth knowing before editing

- "V1" model variants (`*_v1` in `robust_training/configs/model/`, `ares/model/v1_convnext.py`,
  `v1_block.py`) are a biologically-inspired V1 front-end; adversarial training (madry/trades) with
  V1 noise enabled is explicitly unsupported and raises at startup (`v1_noise_mode` must be null).
- "Continuation" runs (resuming a finished/partial model at a new epsilon or norm) resolve their
  starting checkpoint from a dependency model's *DB-recorded best* checkpoint, not just `last.pth.tar`
  — the exact rule (reset-epoch vs resume-epoch) differs per job manager; see each README's
  "checkpoint args" section rather than assuming.
- CSV edits to `csv/*.csv` are picked up live by both job managers before each claim — you generally
  don't need to touch the Python to change a model's hyperparameters or priority, only the CSV row.

## Environment awareness & path portability (important)

This repo runs in three different execution contexts — Botero, the Slurm cluster, and the AIRCC
sandbox — and dataset/checkpoint/output paths differ between them. Never assume one fixed absolute
path; before running or editing jobs, adapt paths to the current machine.

**Detecting execution context:**
- If `SLURM_JOB_ID`/`SLURM_PROCID` is present, assume a Slurm execution context.
- Otherwise assume Botero (local) unless AIRCC-specific env vars (`AIRCC_DB`/`AIRCC_MODEL_ID`) are set.

**Path rules:**
- Always verify and update these path categories before launching: dataset roots (`train_dir`,
  `eval_dir`, `val_dir`), checkpoint/model roots, output/log directories, and repo absolute paths
  (`/home/.../projects/ares`) if scripts rely on them.
- Prefer configurable CLI/Hydra overrides (`dataset.train_dir=...`, `dataset.eval_dir=...`) over
  hardcoded edits when possible.
- Keep project-local relative paths when possible; use absolute paths only when required by cluster
  job scripts.
- Many scripts in `sbatches/`, `sbatches_botero/`, and the job managers contain user-specific
  absolute paths under `/home/ashtomer/...`; adapt them per machine.
- Preserve existing behavior unless the task explicitly asks for path refactoring.

**Known dataset paths in this repo (verify which apply to the current machine):**
- `/storage/test/bml_group/tomerash/datasets/imagenet/train/`
- `/storage/test/bml_group/tomerash/datasets/imagenet/val/`
- `~/datasets/imagenet/train`
- `~/datasets/imagenet/val`
- `/mnt/data/datasets/imagenet_sample/train`
- `/mnt/data/datasets/imagenet_sample/val`
- `/mnt/data/datasets/imagenet/val`

**Files where dataset paths are defined:**
- `robust_training/configs/dataset/imagenet.yaml`
- `robust_training/train_configs/*.yaml`
- `sbatches/botero_tests.sbatch`
- `data_analysis/run_pgd_validation_sweep.py`
- `data_analysis/autoattack_eval.py`
- `data_analysis/PGD_VALIDATION_README.md`
- `data_analysis/FINAL_EVAL_README.md`

**Recommended workflow before running jobs:**
1. Detect environment (Botero / Slurm / AIRCC).
2. Confirm active dataset roots exist on this machine.
3. Override dataset paths via CLI/Hydra instead of editing many files.
4. Confirm output/checkpoint directories are writable.
5. Run.
