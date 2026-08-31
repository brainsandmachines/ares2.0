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

**CSV edits are live.** The manager re-reads and re-validates the CSVs before
*every* claim (never caching them for the process lifetime) and pushes CSV
`priority` / `training.epochs` onto rows that are still pending and unclaimed, so
any model not yet claimed picks up your edit — no need to cancel and resubmit the
sbatch. Models already training keep the row they were launched with. A malformed
or half-written CSV blocks claiming (logged as `CSV RELOAD FAILED`) rather than
silently running stale values; the sync never inserts rows, so seeding stays a
deliberate `seed_db.py` step.

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
| `job_manager.py` | 2 slots/GPU, CSV reload + spec sync before every claim, requeue-before-claim, heartbeat, `--dry-run`. |
| `seed_db.py` | Selective seeder (`--arch`, `--init`) + `--reconcile`. |
| `status.py` | Read-only dashboard. |
| `slurm/job_manager.sbatch` | Pyxis array `1-200%16` on `sandbox`, v2 image. |
| `slurm/smoke_train.sbatch` | One 1-epoch training in the v2 image (launch test). |
| `scripts/backup_aircc_models.sh` | Nightly status check + every-3-days rsync from the sshfs AIRCC mount to Botero. |
| `notify.py` | The one email transport every notifier in the repo uses: SMTP/`mail`, spooling, and the mail archive. |
| `digest.py` | Drains the alert spool into one morning mail (07:30 cron). |
| `mail_log.py` | Reader for the mail archive -- what the notifiers have sent you. |
| `tests/` | DB unit tests + standalone GPU-cleanup test. |

## Alert mail

Every alert -- from `daily_monitor`, the two backup scripts, the QNAP mirror,
`aa_sweep`, the catastrophic-overfitting notifier and the sjm failure analyzer --
goes through `notify.make_emailer(source=...)`, which does three things:

1. **Archives it**, always, to `logs/mail/YYYY-MM.jsonl` (one JSON record per
   mail: `ts`, `msg_id`, `source`, `subject`, `body`, `urgent`, `routed`, and
   `send_error` if the send failed). `ARES_MAIL_ARCHIVE=0` turns this off; a path
   relocates it. This is the durable record -- the spool below is deleted daily.
2. **Spools it** when `ARES_ALERT_SPOOL` is set, so `digest.py` mails a normal
   alert as part of the next morning's digest instead of on its own.
3. **Mails it immediately** when the alert is `urgent=True` (a collapsing run, a
   broken backup), when no spool is configured, or when the spool has gone stale
   (`ARES_ALERT_SPOOL_STALE_HOURS`, default 36 -- i.e. the digest cron died).

```bash
python -m aircc.aircc_job_manager.mail_log                       # last 7 days, one line each
python -m aircc.aircc_job_manager.mail_log --since 30d --source aircc.backup
python -m aircc.aircc_job_manager.mail_log --grep rsync --full   # with bodies
python -m aircc.aircc_job_manager.mail_log --id 4f2a1c9b0e77     # one alert
```

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

`scripts/backup_aircc_models.sh` rsyncs the AIRCC `results/models` (over direct
ssh) to `/mnt/data4t/aircc_archive/models`. Install the cron line shown in the
script header.

It is an **archive, not a mirror**: AIRCC deletes the results tree in late August
2026, so nothing is ever deleted from the destination and every checkpoint is kept
(~1.9TB). That is affordable only because dirs whose DB status is `running` are
skipped — their checkpoints are rewritten every few epochs — and each is picked up
in full on the first pass after it finishes. Failed runs live in a separate
`results/models_failed` tree, archived once by hand. `AIRCC_BACKUP_SKIP_RUNNING=0`
forces a full catch-all sweep.

The cron fires nightly but the two halves run at different cadences: the
aircc-status check runs every night, while the rsync runs at most every
`AIRCC_BACKUP_MIN_INTERVAL_HOURS` (default 72h), paced off `.backup.attempted`.
A full pass moves ~450GB over sshfs at ~3.5MB/s and takes well over a day, so
nightly attempts mostly collided with the previous night's run. On a night with
no rsync the script exits **75**, which tells `daily_monitor --backup-rc` not to
validate a `backup.log` block it did not produce — the DB health checks still
run. Any other non-zero exit is a real backup failure.
