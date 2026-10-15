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
- **No heartbeat-based release.** A SIGTERM trap in the sbatch releases the row on
  time-limit; a fallback releases `running` rows whose owning Slurm job is dead
  (`squeue`/`sacct`) at each task start. An hourly Botero cron
  (`scripts/cron_release.sh` → `cron_release.py`) also frees rows whose owner is not
  training — gone, `CANCELLED`, `COMPLETING`, `PENDING` after a `NODE_FAIL` requeue
  (same job id), or restarted after the claim — and `scancel`s an owner whose
  heartbeat is >6h stale; the row is then freed by the next sweep. It never cancels
  a row in `final_eval`, and cancels none when >2 stall at once (a shared cause).
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
| `classify.py` / `failure_analyzer.py` / `notify.py` | Deterministic failure classification + codex/email escalation. `notify.make_emailer` maps `SJM_SMTP_*` onto the shared `aircc_job_manager.notify` transport (so sjm alerts share the digest and the mail archive), and only falls back to its own SMTP/`mail` sender if that finds none. |
| `monitor.py` | Botero pass: escalate new failure signatures to codex (DB-driven, no ssh). |
| `seed.py` | Upsert / `--reconcile` rows from a CSV (the "add a job" flow). |
| `release.py` | SIGTERM-trap entrypoint: return a row to `pending`. |
| `cron_release.py` / `scripts/cron_release.sh` | Hourly Botero cron sweep (over ssh on the login node): release rows no live task owns, scancel stalled owners. `cron_release.sh --dry-run` previews. |
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

## Re-open a finished model for its AutoAttack grid (eval-only)

Every job ends with the full AutoAttack grid (`final_eval_aatype: full_sweep`: linf/l2/l1 ×
`final_eval_aa_eps_inputs` on best/last/advbest — the trained cell first, so `best_checkpoint`
lands in the DB early, then the rest). A model can still be `finished` without it: it was already
running when a config change landed (a process keeps the config it started with), or its
final_eval raised (logged, exit 0, marked finished anyway). Find and re-open those:

```bash
python -m slurm_job_manager.reopen_eval --db $SJM_DB                  # dry run: list + missing cells
python -m slurm_job_manager.reopen_eval --db $SJM_DB --apply --all    # or --model NAME (repeatable)
```

A re-opened row goes back to `pending` (only `finished` rows with `current_epoch >= total_epochs`;
short rows are `seed --reconcile`'s job). The re-claim resumes from its own `last.pth.tar` at the
final epoch, trains nothing, and final_eval runs only the missing cells. While it is pending/running,
its **not-yet-claimed dependents are blocked again** (the claim gate wants the dependency finished);
already-running dependents are unaffected.

## Verify (offline)

```bash
python -m pytest slurm_job_manager/tests/ -q
```
