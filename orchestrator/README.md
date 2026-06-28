# Orchestrator

Botero-driven, deterministic SQLite orchestrator. The cluster runs a **pull**
worker pool: array tasks atomically claim jobs from the DB and walk each model
through its pipeline. An hourly botero cron only tops up the pools and handles
failures. Replaces the archived push-daemon (`orchestrator/legacy/`).

## How it works

```
BOTERO cron (hourly: python -m orchestrator.monitor)
  per partition [rtx_pro_6000 %6, rtx6000 %8]:
    1. top up   -> if <ORCH_MIN_REMAINING controller tasks left AND claimable work
                   sbatch controller array immediately (--array=1-200%N)
    2. settle   -> sleep 30s if a pool was submitted
    3. capacity -> running tasks == min(capacity, available work)?  (recheck once)
    4. failures -> map FAILED tasks -> DB row; classify deterministically;
                   escalate only unknown signatures to `codex` (deduped, once)
       │  (shared Slurm mount: SQLite DB)
       ▼
CLUSTER controller array task (sbatches/controller.sbatch -> orchestrator.controller)
  1. requeue stale  -> release dead owners (TRAINING>8h, AA_EVAL>36h, PLOTTING>2h)
  2. claim          -> atomic BEGIN IMMEDIATE pick of one eligible row
                       (PLOTTING > AA_EVAL > TRAINING > PENDING, then protocol priority)
  3. run remaining pipeline, in-process:
       train (run_stage.sh)  -> hook sets AA_EVAL
       AA    (run_stage.sh)  -> hook sets PLOTTING
       plot  (plot_autoattack_comparation_orch.py) -> sets FINISHED + best_checkpoint
  4. on failure: classify -> requeue (transient) or mark FAILED
```

The DB (`models_queue`) is the single source of truth. Status is one pipeline
column:

```
PENDING -> TRAINING -> AA_EVAL -> PLOTTING -> FINISHED      (+ FAILED)
```

Ownership invariant: `slurm_job_id IS NULL` ⇔ claimable; non-NULL ⇔ a live array
task owns it. Each row also carries `current_epoch`/`next_epoch` (resume point),
`final_epoch` (target), `requeued` (counter) and `best_checkpoint`.

## Setup (botero)

```bash
cp orchestrator/.env.example orchestrator/.env   # then edit (see open items)
PY=/home/tomer_a/miniconda3/envs/ares/bin/python
$PY -m orchestrator.cli seed                      # seed models_queue from experiments.yaml
$PY -m orchestrator.cli status
$PY -m orchestrator.cli monitor --dry-run         # show top-up/failure decisions, submit nothing
# install the hourly cron:
#   0 * * * * cd ~/Documents/ares && $PY -m orchestrator.monitor >> orchestrator/monitor.log 2>&1
```

To run independently the only recurring action is the hourly `monitor`, which
submits fresh controller pools as the old ones drain.

## CLI

| command | purpose |
|---|---|
| `seed` | upsert rows from `experiments.yaml` (idempotent) |
| `status [--filter STATUS]` | table of all rows (status, epoch, requeued, best ckpt) |
| `validate-model <id>` | DB state + live squeue cross-check (zombie detection) |
| `force-status <id> <STATUS>` | operator override (releases ownership) |
| `requeue <id>` | clear the Slurm owner so the row is claimable again |
| `monitor [--dry-run]` | one top-up + capacity + failure pass |

## Adding experiments

Edit `experiments.yaml`. `model_dir` is auto-derived for small non-v1 models as
`{models_root}/convnext_small_<job_name>` (matches the launchers). Set it
explicitly for v1/base/large. `final_epoch` defaults to warmup+linear+plateau.

## Open items to confirm on first deploy

1. **Shared-mount DB path** — set `ORCH_DB` (botero view) and `ORCH_DB_CLUSTER`
   (cluster view) in `.env`.
2. **SSH alias** (`ORCH_SSH_HOST`) and `ORCH_CLUSTER_USER/REPO/LOGS_ROOT`.
3. **Diagnosis/email transports** — `codex` (override via `ORCH_LLM_CMD`) and
   `mail` are auto-detected; absent, the analyzer still writes a raw-traceback
   report under `recommendations/`.

## Verification

- Offline (botero, `ares` env): `python -m pytest orchestrator/tests/ -q`.
- Cluster (needs Slurm + torch): see `orchestrator/SLURM_TEST_PLAN.md`.
