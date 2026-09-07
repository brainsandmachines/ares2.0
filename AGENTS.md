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
(`data_analysis/`), and a CSV-driven job-queue manager that keeps the Slurm cluster busy running
training campaigns (`slurm_job_manager/`). A second manager, `aircc/aircc_job_manager/`, drove the
now-finished AIRCC allocation; its queue is retired but its shared helpers are still live — see
"AIRCC" below.

## Two systems — always know which one you're targeting

1. **Botero** — this machine. Where all editing, git, and orchestration happens. Never edit code
   through `~/slurm_mount` (a read-only sshfs mount, for inspecting remote logs/checkpoints only).
2. **Slurm cluster** (`ssh slurm`, remote repo `/home/ashtomer/projects/ares`) — run via
   `slurm_job_manager/`. Partitions `rtx_pro_6000` (96GB) / `rtx6000`. One array task = one GPU =
   one model, single-process train→final_eval→plot.

`slurm_job_manager` is the only live queue, and its philosophy is: **the CSV is ground truth** (one
column per Hydra override), the SQLite DB holds only operational state (claim status, progress, best
checkpoint).

### AIRCC — a frozen archive and a shared library, not a system

The AIRCC B200 allocation is **finished**. This machine no longer talks to it: there is no live DB
and no job submission, so never propose an AIRCC run, requeue, `sbatch`, or `ssh aircc` command.
Two things survive it, and both are load-bearing:

- **The frozen archive** — `/mnt/botero/aircc_archive` on the QNAP, including
  `aircc_jobs_final_latest.sqlite` (127 finished of 323 rows, byte-for-byte the set the live DB last
  reported). `aa_sweep`'s Botero lane reads the finished-model *list* from it (`aa_sweep/config.py`);
  the checkpoints themselves come from `/mnt/data4t/models`, never from the archive. See
  `aa_sweep/README.md` § "The AIRCC side is a list of names, nothing more".
- **Shared library code under `aircc/aircc_job_manager/`** — retired as a queue, live as a library:
  - `notify.py` (`make_emailer`) is the repo-wide mailer, with seven callers outside `aircc/`:
    `slurm_job_manager/notify.py`, `aa_sweep/submit.py` and its two cron scripts,
    `scripts/mirror_archives_to_qnap.sh`, `data_analysis/catastrophic_overfitting_notifier.py`,
    and `slurm_job_manager/scripts/{check_slurm_status,backup_slurm_models}.sh`. A daily
    `python -m aircc.aircc_job_manager.digest` cron still mails the spooled alerts.
  - `progress.py` is imported unconditionally by `ares/utils/train_loop.py` and
    `robust_training/adversarial_training.py`. `slurm_job_manager`'s DB schema is a superset of
    AIRCC's, so those in-training hooks work unchanged against either DB (the only cross-package
    import between the two managers).
  - `best_checkpoint.py` is what `model_store/build_experiments.py` uses to resolve the blessed
    checkpoint per threat model.

  So don't delete, rename, or "clean up" `aircc/aircc_job_manager/`: relocating those helpers is a
  ten-call-site refactor, not a tidy-up. Its queue-driving parts (`job_manager.py`, `seed_db.py`,
  `generate_csvs.py`, `daily_monitor.py`, `status.py`, `csv/`) are dead but harmless — treat them
  like `archive/` below: reference only, don't extend.

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
```
Seeding/submitting new campaign rows or sbatch submissions are not side-effect-free — confirm with
the user before running `seed.py` writes or `sbatch`.

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
  `AIRCC_MODEL_ID` env vars are set. Those env-var names are historical: `slurm_job_manager` sets
  them to point at its *own* DB, so they say nothing about which cluster you are on.
- **`robust_training/configs/`** — Hydra config groups (`model/`, `training/`, `dataset/`,
  `optimizer/`, `attacks/`, `epsilon_schedule/`, `continuation/`, `checkpointing/`, `machine/`,
  `lr_scheduler/`, `dist/`); `config.yaml` is the composing default.
- **`slurm_job_manager/`** — the live CSV+SQLite job queue (see its README for the file-by-file
  breakdown: `db.py` atomic claim, `lifecycle.py` command-building + subprocess run, `controller.py`
  main loop, `csv_spec.py` column schema, `status.py` dashboard, `seed.py` to add jobs).
  **`aircc/aircc_job_manager/`** is its retired twin — same shape, kept for the shared helpers
  described under "AIRCC" above. Don't assume behavior is shared between the two beyond the DB
  schema/progress-hook overlap noted there.
- **`data_analysis/`** — standalone eval/plotting scripts (`final_eval.py`, `autoattack_eval.py` and
  variants, `training_plots.py`, shape-bias analysis) generally run against already-trained
  checkpoints, independent of the job managers.
- **`aa_sweep/`** — nightly driver that completes the AutoAttack grid (3 norms × 5 eps × 3
  checkpoint kinds) for every finished model. **Two independent lanes that never exchange models:**
  the Slurm lane submits sbatch jobs against the cluster's own `results/models` copies, and the
  Botero lane runs on this machine's RTX 4090 against `/mnt/data4t/models`. Each writes its result
  CSVs back into the model dir it evaluated; the split is by cluster-directory presence and is
  disjoint by construction (`plan.build_plan`). Nothing in the package copies a checkpoint or a
  result between machines — propagation is the weekly rsync's job (see `model_store/` below). If a
  change here seems to need an rsync/scp/`scancel`, it is the wrong change. Read
  `aa_sweep/README.md` first.
- **`model_store/`** — owns the curated model trees on Botero: `/mnt/data4t/models` (every model,
  keeper checkpoints only) and `/mnt/data4t/models_for_experiments`
  (`<arch>/<protocol>/<norm>/<name>.pth.tar` symlinks to the blessed checkpoint — the root
  `epsilon_bounded_contstim` reads). The QNAP (`/mnt/botero/{aircc,slurm}_archive`) stays the
  read-only master. Models reach Botero by **two independent weekly rsync routes**:
  route 1 `slurm_job_manager/scripts/backup_slurm_models.sh` (Sun 09:00, slurm → QNAP) and
  route 2 `model_store/scripts/ms_weekly_sync.sh` (Mon 09:00, QNAP → `/mnt/data4t/models` →
  the experiment symlinks). Neither may be folded back into the other: route 1 must not be
  marked failed by an index rebuild, and route 2 must be re-runnable without re-pulling
  ~0.9 TB. Run individual passes via `model_store/scripts/ms_run.sh` (dry-run by default, one
  tmux session each, logs in `slurm_job_manager/logs/reorg/`). Read `model_store/README.md`
  before touching it — in particular: **mtime is unreliable here** (70 AIRCC
  `model_best.pth.tar` files share a bulk-rewrite mtime, so content hash decides, and
  checkpoints are chosen by epoch not time), **never add `--inplace`/`--append`** to an rsync
  whose destination is under `models/` (it would write through a hardlink into the archive),
  and **never mirror `models_for_experiments` to the QNAP** — CIFS `nounix` cannot create a
  symlink and returns EIO on every one. `nlink == 1` identified a discard under the old
  hardlink-only build; post-cutover models arrive as real copies, so that test now needs the
  Step 3 caveat in the README. Nothing in the package deletes; removals are a `mv` into
  `pending_deletion/<date>/`.
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

This repo runs in two execution contexts — Botero and the Slurm cluster — and dataset/checkpoint/
output paths differ between them. Never assume one fixed absolute path; before running or editing
jobs, adapt paths to the current machine. (A third context, the AIRCC sandbox, is retired: paths
under `/shared/cycle2_bgu_golan_prj/...` in older scripts and logs are dead.)

**Detecting execution context:**
- If `SLURM_JOB_ID`/`SLURM_PROCID` is present, assume a Slurm execution context.
- Otherwise assume Botero (local). Note that `AIRCC_DB`/`AIRCC_MODEL_ID` do **not** indicate AIRCC —
  `slurm_job_manager` sets them on every run to point at its own DB.

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

**Botero model/checkpoint roots (owned by `model_store/`, see its README):**

| path | role |
|---|---|
| `/mnt/botero/{aircc,slurm}_archive` | QNAP master archive, append-only. **Read-only** — never write here outside the two backup scripts. |
| `/mnt/data4t/models` | curated working copy: all models, keeper checkpoints only, no `checkpoint-N`. Hardlinked, so it consumes ~no space. |
| `/mnt/data4t/models_for_experiments` | `<arch>/<protocol>/<norm>/<name>.pth.tar` symlinks to the blessed checkpoint; what `epsilon_bounded_contstim` loads. |
| `/mnt/data4t/pending_deletion/<date>` | staged for the user to erase by hand. Never `rm` anything here on their behalf. |
| `/mnt/data/models` | third-party `robustness`-library zoo (`resnet50_l2_eps*.ckpt`). **Not** ares output; leave alone. |

The `/mnt/data4t/{aircc,slurm}_archive` trees are the *old* layout, superseded by
`/mnt/data4t/models`; `/mnt/data/robustness_models` is the *old* promoted zoo, superseded by
`models_for_experiments`.

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
1. Detect environment (Botero / Slurm).
2. Confirm active dataset roots exist on this machine.
3. Override dataset paths via CLI/Hydra instead of editing many files.
4. Confirm output/checkpoint directories are writable.
5. Run.
