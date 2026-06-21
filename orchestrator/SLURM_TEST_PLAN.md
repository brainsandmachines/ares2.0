# Orchestrator rebuild — cluster test plan (run when Slurm is back)

Everything offline is already covered by `python -m pytest orchestrator/tests/`
(58 tests, run under the `ares` conda env on botero). The checks below need a
live cluster (Slurm + GPUs + torch) and so were **not** runnable during the
rebuild. Run them in order; each lists the exact command and the pass criterion.

Prereqs:
- Code synced to the cluster: `git push` on botero, then
  `ssh slurm "cd /home/ashtomer/projects/ares && git pull"`.
- `.env` points `ORCH_DB`/`ORCH_DB_CLUSTER` at the shared DB.
- `PY=/home/tomer_a/miniconda3/envs/ares/bin/python` (botero CLI).
- Seed/import some rows: `$PY -m orchestrator.cli seed` (or `import_existing`).

---

## 1. Controller array submits and respects the concurrency cap

```bash
ssh slurm "cd /home/ashtomer/projects/ares && \
  sbatch --partition=rtx_pro_6000 --array=1-200%6 \
         --export=ALL,ORCH_DB=/home/ashtomer/projects/ares/orchestrator/orchestrator.db \
         sbatches/controller.sbatch"
ssh slurm "squeue -u ashtomer -r -o '%i|%P|%T|%j' | grep orch-controller | grep RUNNING | wc -l"
```
**Pass:** at most 6 controller tasks RUNNING on `rtx_pro_6000` at once (8 for an
`%8` pool on `rtx6000`). Each running task holds exactly one GPU.

## 2. Atomic claim under real concurrency (no double-claim)

Seed N≥14 PENDING rows, submit pools on both partitions, let them run.
```bash
$PY -m orchestrator.cli status            # watch status transitions
# After tasks have claimed:
ssh slurm "sqlite3 .../orchestrator.db \
  'SELECT slurm_job_id, COUNT(*) FROM models_queue \
   WHERE slurm_job_id IS NOT NULL GROUP BY slurm_job_id HAVING COUNT(*)>1'"
```
**Pass:** the query returns nothing (no Slurm task owns two rows; the
`BEGIN IMMEDIATE` claim serialized all array tasks across both partitions).

## 3. Per-epoch DB writes (epoch+1) and the train→AA→plot handoffs

Pick one claimed TRAINING model and watch its row.
```bash
watch -n 30 "$PY -m orchestrator.cli validate-model <model_id>"
```
**Pass, in order:**
- `current_epoch` increments each epoch and equals `next_epoch` (exact epoch+1).
- at the final epoch the status flips **TRAINING → AA_EVAL** (training hook).
- after the AutoAttack sweep status flips **AA_EVAL → PLOTTING** (AA hook).
- after plotting status is **FINISHED** with `best_checkpoint` set, and
  `autoattack_eval_comparation_<model>.png` exists in the model dir.

## 4. best_checkpoint correctness

For a finished `..._l2_2_init1` model:
```bash
$PY -m orchestrator.cli validate-model <model_id>   # note best_ckpt
# cross-check against the CSVs:
#   best  -> autoattack_sweep_results.csv
#   last  -> autoattack_sweep_results_last.csv
#   advbest -> autoattack_sweep_results_advbest.csv
```
**Pass:** `best_checkpoint` is the kind with the highest `robust_acc` at
`attack_norm=l2, epsilon_input=2`. For a baseline/clean model it is the kind
with the highest `clean_acc`.

## 5. Resume-from-next_epoch after an interruption

`scancel` a RUNNING training task mid-run.
```bash
ssh slurm "scancel <task_id>"
# wait for requeue (either the >8h scan, or force it):
$PY -m orchestrator.cli requeue <model_id>          # clears the dead owner now
# next controller pool claims it:
$PY -m orchestrator.cli validate-model <model_id>
```
**Pass:** the row becomes claimable (`slurm_job_id` NULL, `requeued` bumped),
is re-claimed, and training **resumes from `next_epoch`** (not from 0) — confirm
the first logged epoch ≈ the pre-cancel `next_epoch`.

## 6. Stale-requeue thresholds + liveness guard

- Confirm a healthy RUNNING task is **not** requeued (its `slurm_job_id` is still
  active via `sacct`), even past a short idle gap.
- Confirm a dead/cancelled owner past its threshold (TRAINING 8h / AA_EVAL 36h /
  PLOTTING 2h) **is** requeued by the next controller's stale scan.

**Pass:** alive owners survive; dead owners past threshold are released exactly
once (`requeued` increments by 1, not repeatedly).

## 7. Hourly monitor: top-up with afterany dependency, no double-queue

Let a pool drain below `ORCH_MIN_REMAINING` (20) tasks, then:
```bash
$PY -m orchestrator.cli monitor          # one real pass (submits if warranted)
ssh slurm "squeue -u ashtomer -r -o '%i|%T|%j|%r' | grep orch-controller"
```
**Pass:**
- a new pool is submitted **per partition** with `--array=1-200%6` (pro) /
  `%8` (rtx6000) and `--dependency=afterany:<latest pool id>`.
- the dependent pool stays PENDING with reason `Dependency` until the prior pool
  finishes; a second `monitor` run does **not** queue another (double-queue
  guard), and tops up nothing while ≥20 tasks remain.

## 8. Capacity check

With claimable work available and a partition under cap, run `monitor` and read
the log.
**Pass:** it logs the running-vs-expected comparison and (if a pool was just
submitted) re-checks after 30s rather than thrashing.

## 9. Failure handling: deterministic classes vs codex escalation

- Induce a **transient** failure (e.g. `scancel ... ` / time limit): monitor maps
  the FAILED task to its row and **requeues** it (no codex).
- Induce a **deterministic** error (e.g. a bad Hydra override → ModuleNotFound /
  ValueError): monitor marks the row **FAILED**, writes a raw report under
  `recommendations/`, no codex.
- Induce a **novel** traceback: monitor classifies it `unknown` and escalates to
  `codex exec` **once**; a repeat of the same signature is **deduped** (no second
  codex call), verified via the `failure_hashes` table and a single `.md` report.

**Pass:** each class takes its deterministic action; only the unknown signature
invokes `codex`, exactly once.

---

## Notes / things to watch on first real run
- `sbatches/run_stage.sh` reuses `train_launcher_lib.sh::parse_train_job`, so the
  training command matches the legacy `golan-trainmodels.sbatch` exactly — diff a
  dry echo of `${cmd[*]}` against the old launcher for one model if unsure.
- The old push launchers (`golan-trainmodels.sbatch`, `epsilon_curriculum.sbatch`,
  `stage_runner.sh`) are superseded by `controller.sbatch` + `run_stage.sh`; don't
  submit them for orchestrated rows.
- `codex exec` runs read-only diagnosis; confirm it's logged-in on botero
  (`codex login` / `codex doctor`) so escalation isn't silently skipped.
