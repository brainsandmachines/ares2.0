# Orchestrator

Botero-driven, deterministic SQLite orchestrator that keeps the cluster GPUs
full and walks each model through a stage pipeline. Replaces the archived
`job_manager/` (see `archive/legacy_job_manager_2026-06-19/`).

## How it works

```
BOTERO (daemon, every ORCH_INTERVAL_MINUTES)
  ssh squeue   -> count my running/pending jobs per partition
  reconcile    -> detect finished/failed RUNNING rows
  top up       -> ssh sbatch for each free GPU (6 rtx_pro_6000 + 8 rtx6000)
  failures     -> MD5-fenced LLM analyzer (escalate each signature once)
       │  (shared Slurm mount: SQLite DB)
       ▼
CLUSTER job (golan-trainmodels / epsilon_curriculum)
  stage_runner.sh reads its DB row and runs:
    TRAIN (train-only) -> AA_SWEEP (AutoAttack best,last,advbest; no PGD)
      -> PLOT_RESULTS (compare checkpoints) -> COMPLETED
  training writes atomic per-epoch current_epoch (orchestrator/progress.py)
```

The DB (`models_queue`) is the single source of truth. Each row carries the
launcher, job_name, model_dir, curriculum epoch split (warmup/linear/plateau),
status, current_stage, current_epoch and the last failure signature.

## Setup (botero)

```bash
cp orchestrator/.env.example orchestrator/.env   # then edit (see open items)
PY=/home/tomer_a/miniconda3/envs/ares/bin/python
$PY -m orchestrator.cli seed                      # seed models_queue from experiments.yaml
$PY -m orchestrator.cli status
$PY -m orchestrator.cli tick --dry-run            # show what WOULD be submitted
# install the watchdog:
#   */20 * * * * /home/tomer_a/Documents/ares/orchestrator/cron_watchdog.sh >> .../cron.log 2>&1
```

The daemon (`python -m orchestrator.daemon`) is started/kept-alive by the cron
watchdog; the 20-min cron is the backup, the daemon is the immediate-relaunch
loop.

## CLI

| command | purpose |
|---|---|
| `seed` | upsert rows from `experiments.yaml` (idempotent) |
| `status [--filter STATUS]` | table of all rows |
| `validate-model <id>` | DB state + live squeue cross-check (zombie detection) |
| `force-stage <id> <STAGE> [--status ...] [--kinds ...]` | operator override |
| `tick [--dry-run]` | one scheduling pass |
| `requeue-stale` | reset stuck RUNNING rows to PENDING |

## Adding experiments

Edit `experiments.yaml`. `model_dir` is auto-derived for small non-v1 models as
`{models_root}/convnext_small_<job_name>` (matches the launchers). Set it
explicitly for v1/base/large.

## Open items to confirm on first deploy

1. **Shared-mount DB path** — set `ORCH_DB` (botero view) and `ORCH_DB_CLUSTER`
   (cluster view) in `.env`. Run `orchestrator/tests` and the NFS lock check
   before trusting per-epoch writes.
2. **SSH alias** (`ORCH_SSH_HOST`) and `ORCH_CLUSTER_USER/REPO/LOGS_ROOT`.
3. **LLM/email transports** — `claude` CLI and `mail` are auto-detected; absent,
   the analyzer still writes a raw-traceback report under `recommendations/`.

## Cluster-side verification (cannot run on botero — needs SLURM + torch)

- Submit one orchestrated job; confirm `current_epoch` advances in the DB,
  then AA CSVs (best/last/advbest) appear, then
  `autoattack_eval_comparation_*.png`, then the row is `COMPLETED`.
- Re-running a `COMPLETED` row exits in seconds (fast inspection).
