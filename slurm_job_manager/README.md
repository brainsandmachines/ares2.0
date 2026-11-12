# Slurm Job Manager (Botero cluster)

A simple, adaptive CSV-driven job manager for the Botero cluster
(`rtx_pro_6000` / `rtx6000`). Replaces the retired `orchestrator/` (left in place,
unused). Same philosophy as `aircc/aircc_job_manager/` — **CSV is ground truth**,
the SQLite DB holds operational state only — but simplified to Botero's rules:

- **One array task = one GPU = one model, training only.** A task requeues dead
  owners, claims one eligible `pending` row, runs a single-process
  `adversarial_training` with `final_eval=False`, then exits. If a model isn't
  trained at the Slurm time-limit, the row goes back to `pending` and the next
  task resumes it from `last.pth.tar`.
- **Eval lane (since 2026-11-12).** When training ends the row goes `running → eval`
  and the task sbatches one **`sjm-eval`** job on the shared **`main`** partition
  (`slurm/manager_eval_main.sbatch`). That job runs the same controller with
  `--lane eval`: it claims one `eval` row and runs the same command, which resumes
  at the final epoch (0 epochs train), so only the config-default `final_eval`
  (full AutoAttack grid, trained cell first → DB `best_checkpoint`/`best_score`,
  then the rest + plot) and shape-bias run → `finished`. The rtx_pro_6000 cards
  never run AutoAttack. Dependents still wait for `finished`. See "Eval lane" below.
- **No heartbeat-based release.** A SIGTERM trap in the sbatch releases the row on
  time-limit; a fallback releases `running` rows whose owning Slurm job is dead
  (`squeue`/`sacct`) at each task start. An hourly Botero cron
  (`scripts/cron_release.sh` → `cron_release.py`) also frees rows whose owner is not
  training — gone, `CANCELLED`, `COMPLETING`, `PENDING` after a `NODE_FAIL` requeue
  (same job id), or restarted after the claim — and `scancel`s an owner whose
  heartbeat is >6h stale; the row is then freed by the next sweep. It never cancels
  a row in `final_eval` as a stall, and cancels none when >2 stall at once (a shared
  cause). The same sweep runs the eval-lane **handoff** and **top-up** (below).
  A released row that has trained every epoch goes to `eval`, not `pending`.
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
| `controller.py` | Cluster worker: requeue dead → claim one → run → trained/finish/requeue/fail. `--lane train` (default) or `--lane eval` (`$SJM_LANE`); `--dry-run` shows commands. |
| `eval_lane.py` | `top_up`: promote stray fully-trained `pending` rows to `eval`, then sbatch `unowned eval rows − PENDING sjm-eval jobs` eval jobs. |
| `classify.py` / `failure_analyzer.py` / `notify.py` | Deterministic failure classification + codex/email escalation. `notify.make_emailer` maps `SJM_SMTP_*` onto the shared `aircc_job_manager.notify` transport (so sjm alerts share the digest and the mail archive), and only falls back to its own SMTP/`mail` sender if that finds none. |
| `monitor.py` | Botero pass: escalate new failure signatures to codex (DB-driven, no ssh). |
| `seed.py` | Upsert / `--reconcile` rows from a CSV (the "add a job" flow). |
| `release.py` | SIGTERM-trap entrypoint: return a row to `pending`. |
| `cron_release.py` / `scripts/cron_release.sh` | Hourly Botero cron sweep (over ssh on the login node): release rows no live task owns, scancel stalled owners. `cron_release.sh --dry-run` previews. |
| `status.py` | Read-only dashboard. |
| `slurm/manager_<partition>.sbatch` | Per-partition training array + env select + SIGTERM-trap release. |
| `slurm/manager_eval_main.sbatch` | One `sjm-eval` job on `main` (eval lane): env + AA batch by card, SIGTERM-trap release. |

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

## Eval lane (`sjm-eval` on `main`)

Row states: `pending → running (train) → eval → running (eval job) → finished`.

- **Launch.** After a clean training exit the controller calls `db.mark_trained` (→ `eval`)
  and `eval_lane.top_up`, which sbatches `slurm/manager_eval_main.sbatch` from the compute
  node — enough jobs to cover every unowned `eval` row not already matched by a PENDING
  `sjm-eval` job. A surplus job (two callers racing) claims nothing and exits in seconds.
- **The job.** `--partition=main --gpus=1`, constraint `rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000`,
  same excludes as the aa_sweep `main` array (pheno nodes, no-`/groups` nodes), 7 days. The
  cluster's submit plugin reroutes it to the **`gpu`** partition (QOS `gpu-part`, **max 5 GPUs per
  user**, shared with any other `--gpus` job you run there — e.g. epsilon_bounded_contstim's
  `pe_*` cells), so eval jobs queue as `QOSMaxGRESPerUser` while those hold the 5. AA batch
  by card, always the same 1024 images: ≥45 GB keeps the config's 128 × 8, smaller cards get
  `final_eval_aa_batch_size=32 final_eval_aa_max_batches=32` (via `$SJM_EXTRA_OVERRIDES`).
  Nothing else is new: it is `controller --lane eval` running `lifecycle.build_command` (own
  `last.pth.tar` → 0 epochs → `final_eval` → shape-bias); the in-process hook writes
  `best_checkpoint`/`best_score` exactly as before. Logs: `outs/sjm_eval/<jobid>.out`.
- **Time limit / preemption.** The TERM trap releases the row back to `eval` (it is fully
  trained); the next `sjm-eval` job resumes the grid where the CSVs stop.
- **Catch-up.** The hourly `cron_release` sweep re-runs `top_up`, so a failed submit (or a
  dead eval job) is covered within the hour; its summary line reports
  `eval_unowned=` / `eval_submitted=`. By hand: `sbatch slurm_job_manager/slurm/manager_eval_main.sbatch`.
- **Handoff of old-config runs.** A `running` row at `current_epoch >= total_epochs` whose owner
  is on `rtx_pro_6000`/`rtx6000`, with its last epoch write >30 min old, is an old-config process
  doing AutoAttack in-process: the sweep `scancel`s it, the trap releases the row to `eval`, and
  the eval lane redoes only the missing cells. Owners on `main` are never handed off.
- **Inspect.** `python -m slurm_job_manager.status` lists the `eval` queue;
  `python -m slurm_job_manager.controller --lane eval --dry-run` shows the eval commands;
  `squeue -u $USER -n sjm-eval`.

## Re-open a finished model for its AutoAttack grid (eval-only)

Every model gets the full AutoAttack grid in its eval job (`final_eval_aatype: full_sweep`:
linf/l2/l1 × `final_eval_aa_eps_inputs` on best/last/advbest — the trained cell first, so
`best_checkpoint` lands in the DB early, then the rest). A model can still be `finished` without
it: it trained under an older config (a process keeps the config it started with), or its
final_eval raised (logged, exit 0, marked finished anyway). Find and re-open those:

```bash
python -m slurm_job_manager.reopen_eval --db $SJM_DB                  # dry run: list + missing cells
python -m slurm_job_manager.reopen_eval --db $SJM_DB --apply --all    # or --model NAME (repeatable)
```

A **daily Botero cron (07:00)** runs this with `--apply --all` over ssh:
`slurm_job_manager/scripts/reopen_eval_daily.sh` (`--dry-run` to preview; log in
`logs/reopen_eval.log`), and mails when it re-opens or caps something. It never re-opens models in
`aa_sweep/sweep_excludes.csv` or ones the aa_sweep queue still holds, and `--all` stops after
`--max-reopens` (2) re-opens of the same model — those show up as `capped` (an eval that keeps
failing: read its final_eval log). `--model NAME` ignores the cap.

A re-opened row goes back to `eval` (only `finished` rows with `current_epoch >= total_epochs`;
short rows are `seed --reconcile`'s job) and `--apply` queues `sjm-eval` jobs for it. The eval job
resumes from its own `last.pth.tar` at the final epoch, trains nothing, and final_eval runs only
the missing cells. While it is eval/running,
its **not-yet-claimed dependents are blocked again** (the claim gate wants the dependency finished);
already-running dependents are unaffected.

## Verify (offline)

```bash
python -m pytest slurm_job_manager/tests/ -q
```
