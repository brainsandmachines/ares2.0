# AIRCC Job Manager

A self-contained, CSV-driven job manager for the AIRCC B200 (`sandbox`) allocation.
It keeps up to 16 GPUs busy, runs **2 training lifecycles per GPU**, and on each
slot free immediately claims the next eligible model. Each lifecycle is a **single**
`python -m robust_training.adversarial_training` run that — via the config default
`final_eval=True` — trains, then evaluates 3 checkpoints (best/last/advbest) and
writes the comparison plot, all in one process under `results/models/`.

The **CSVs in `csv/` are the ground truth**. Each row has **one column per Hydra
override** (header = the Hydra key); the manager builds the command from the
non-empty override cells. The SQLite DB holds **operational state only** (no
hyperparameters): claiming, progress, best checkpoint.

## Files

| File | Purpose |
|---|---|
| `generate_csvs.py` | Deterministic generator for the 3 arch CSVs (312 rows each), column-per-override. |
| `csv/{convnext_small,base,large}.csv` | Permanent model specs (committed; never edited by the manager). |
| `csv_spec.py` | `METADATA_COLUMNS`, `OVERRIDE_COLUMNS`, `build_overrides(row)`, loaders. |
| `db.py` | `AirccDB`: `BEGIN IMMEDIATE` claim w/ dependency gate, stale requeue, `reconcile`, progress/best-ckpt writes. |
| `progress.py` | In-training DB hooks: `update_epoch`, `heartbeat`, `write_best_checkpoint` (no-op unless `AIRCC_DB`+`AIRCC_MODEL_ID`). |
| `best_checkpoint.py` | Score best/last/advbest by an explicit (norm,eps) threat model. |
| `lifecycle.py` | Build the command from columns + resolve checkpoints; run one subprocess; mark finished/failed. |
| `job_manager.py` | 2 slots/GPU, requeue-before-claim, heartbeat, `--dry-run`. |
| `seed_db.py` | Selective seeder (`--arch`, `--init`) + `--reconcile`. |
| `status.py` | Read-only dashboard. |
| `slurm/job_manager.sbatch` | Pyxis array `1-200%16` on `sandbox`, v2 image. |
| `slurm/smoke_train.sbatch` | One 1-epoch training in the v2 image (launch test). |
| `scripts/backup_aircc_models.sh` | Nightly rsync from the sshfs AIRCC mount to Botero. |
| `tests/` | DB unit tests + standalone GPU-cleanup test. |

## CSV columns

- **Metadata** (manager only, not overrides): `model_name`, `arch`, `init`,
  `protocol`, `init_mode`, `epoch_variant`, `dependency_model_name`, `threat_norm`,
  `threat_eps`, `priority`, `notes`.
- **Overrides** (header = Hydra key; emitted as `key=value` when non-empty):
  `model`, `model.experiment_name/num/v1_noise_mode/compile_model`, `output_dir`,
  `training.epochs/batch_size`, `attacks.*`, `dataset.dvd.*`, `continuation.*`,
  `checkpointing.save_best_adv`, `epsilon_schedule.*`.

The launcher appends the dynamic checkpoint args and `+machine=aircc`:
- scratch → `model.resume=<own last>`
- continuation (`resetepoch`/madry/trades/v1) → `continuation.checkpoint_path=<dep DB-best>` + `model.resume=<own last>`
- resume (`contepoch`/gradnorm) → `model.resume=<own last else dep DB-best>`

**All continuation jobs init from the dependency's DB-best checkpoint** — the
best/last/advbest kind that wins by the source's eps+norm robustness.

Per-protocol batch size: 512 (madry, baseline, dvd_baseline, dvd_madry), 256
(trades, gradnorm, dvd_trades, v1). `model.compile_model` only on madry & dvd_madry.
`checkpointing.save_best_adv` on all but the two baselines.

## Quick start

```bash
python -m aircc.aircc_job_manager.generate_csvs          # (re)generate CSVs
# review csv/convnext_base.csv, THEN seed:
export AIRCC_DB=/shared/cycle2_bgu_golan_prj/ashtomer/ares/aircc/aircc_job_manager/aircc_jobs.sqlite
python -m aircc.aircc_job_manager.seed_db --db "$AIRCC_DB" --arch convnext_base --init 1
python -m aircc.aircc_job_manager.job_manager --dry-run   # inspect exact commands
sbatch aircc/aircc_job_manager/slurm/job_manager.sbatch   # launch (fill in $AIRCC_IMAGE)
python -m aircc.aircc_job_manager.status --db "$AIRCC_DB"  # monitor
```

Raised a model's epochs after it finished? Re-open it:
`seed_db --db "$AIRCC_DB" --arch convnext_base --init 1 --reconcile`.

## Running on AIRCC (Pyxis + v2 image)

Jobs run inside the **built** `aircc/Dockerfile` image (NGC `pytorch:25.10-py3` +
timm/hydra/autoattack/dvd). Build it once and import to an enroot squashfs on the
shared mount, then point the sbatch at it:

```bash
docker build -t ares-train:v2 -f aircc/Dockerfile aircc/
enroot import -o /shared/cycle2_bgu_golan_prj/ashtomer/images/ares-train-v2.sqsh dockerd://ares-train:v2
# sbatch already defaults --container-image to that path; override with
#   sbatch --container-image=<other> slurm/job_manager.sbatch
```

`+machine=aircc` supplies `dataset.train_dir/eval_dir`. Each task pins 1 GPU and
runs 2 procs, each capped to `AIRCC_MEM_FRACTION=0.47` of the B200. wandb runs go
to project **adv_train_aircc** (`WANDB_PROJECT`, set per-run by the lifecycle).

## DB write points (ordering: upsert → requeue → claim)

`seed_db.upsert_pending`/`--reconcile` (pre-array) · `requeue_stale` (manager
startup + loop, before any claim) · `claim_next` (BEGIN IMMEDIATE) ·
`progress.update_epoch` (epoch end) · `progress.heartbeat` (log interval) ·
manager heartbeat (AA-eval phase) · `progress.write_best_checkpoint` (after the
final_eval plot) · `mark_finished` / `mark_failed`.

## Backup

`scripts/backup_aircc_models.sh` rsyncs the AIRCC `results/models` (via the local
sshfs mount) to `/mnt/data/robustness_models/aircc_models` nightly. Install the
cron line shown in the script header.
