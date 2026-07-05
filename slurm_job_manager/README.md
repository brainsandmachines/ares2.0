# Slurm Job Manager (Botero cluster)

A simple, adaptive CSV-driven job manager for the Botero cluster
(`rtx_pro_6000` / `rtx6000`). Replaces the retired `orchestrator/` (left in place,
unused). Same philosophy as `aircc/aircc_job_manager/` — **CSV is ground truth**,
the SQLite DB holds operational state only — but simplified to Botero's rules:

- **One array task = one GPU = one model.** A task requeues dead owners, claims
  one eligible row, runs a **single-process** `adversarial_training` (train →
  `final_eval` AutoAttack → plot, all in one process), then exits. If a model
  isn't finished at the Slurm time-limit, the row goes back to `pending` and the
  next task resumes it from `last.pth.tar`.
- **No heartbeat-based requeue.** A SIGTERM trap in the sbatch releases the row on
  time-limit; a fallback releases `running` rows whose owning Slurm job is dead
  (`squeue`/`sacct`). Heartbeat is display-only.
- **Test lane.** Rows flagged `is_test` are claimed *before* production rows.
- **One model per GPU** — no MPS / memory fractions. Clean single-GPU runs.
- **Batch size** = the CSV's `training.batch_size` (the full 96 GB rtx_pro_6000
  value); the rtx6000 sbatch sets `SJM_BATCH_DIVISOR=2` to halve it, rtx_pro_6000
  uses it as-is.
- **Failures:** the controller marks real errors FAILED with the log tail +
  signature; the botero `monitor` escalates each new signature to `codex` once.

## Files

| File | Purpose |
|---|---|
| `csv/*.csv` | Model specs (ground truth); one column per Hydra override + metadata. |
| `csv_spec.py` | Column schema, `build_overrides`, `deps_map`, `is_test_row`. |
| `db.py` | `JobDB`: atomic claim (test lane + dep gate), liveness requeue, `release`, failure-hash dedup. |
| `lifecycle.py` | Build the command from a row (+resume/continuation ckpt) and run one subprocess; stream + return `(rc, tail)`. |
| `controller.py` | Cluster worker: requeue dead → claim one → run → finish/requeue/fail. `--dry-run` shows commands. |
| `classify.py` / `failure_analyzer.py` / `notify.py` | Deterministic failure classification + codex/email escalation. |
| `monitor.py` | Botero pass: escalate new failure signatures to codex (DB-driven, no ssh). |
| `seed.py` | Upsert / `--reconcile` rows from a CSV (the "add a job" flow). |
| `release.py` | SIGTERM-trap entrypoint: return a row to `pending`. |
| `status.py` | Read-only dashboard. |
| `slurm/manager_<partition>.sbatch` | Per-partition array + env select + SIGTERM-trap release. |

The DB schema is a **superset of the aircc `jobs` table**, so the already-live
in-training hooks in `aircc/aircc_job_manager/progress.py` write epoch /
heartbeat / best-checkpoint into it unchanged — the lifecycle just exports
`AIRCC_DB` / `AIRCC_MODEL_ID`. This is the only cross-package import.

## Add a job (2 steps)

```bash
# 1. edit the CSV (add a row, or raise an existing row's training.epochs)
$EDITOR slurm_job_manager/csv/convnext_small.csv
# 2. make the DB aware of it (idempotent; --reconcile re-opens finished rows)
python -m slurm_job_manager.seed --db $SJM_DB --arch convnext_small --reconcile
```

A **test job** is just a row with `is_test=1` (claimed ahead of production).

## Run

```bash
export SJM_DB=/home/ashtomer/projects/ares/slurm_job_manager/jobs.sqlite
# seed the campaign (selective by --init; omit --init for all rows in the CSV)
python -m slurm_job_manager.seed --db $SJM_DB --arch convnext_small --init 1 --init 2
python -m slurm_job_manager.controller --db $SJM_DB --dry-run   # inspect commands

# launch per partition; %K = number of GPUs to occupy (resize + resubmit to scale)
ssh slurm 'cd ~/projects/ares && sbatch slurm_job_manager/slurm/manager_rtx_pro_6000.sbatch'
ssh slurm 'cd ~/projects/ares && sbatch slurm_job_manager/slurm/manager_rtx6000.sbatch'

python -m slurm_job_manager.status --db $SJM_DB                 # monitor
python -m slurm_job_manager.monitor                            # (botero) escalate failures
```

Change occupied GPUs: `scancel` the array and resubmit with a new `--array=1-200%K`.

## Verify (offline)

```bash
python -m pytest slurm_job_manager/tests/ -q
```
