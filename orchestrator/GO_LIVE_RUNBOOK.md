# Orchestrator go-live runbook (push → pull cutover)

The pull-model code is committed on `orchestrator-rebuild` and validated on the
live cluster (see `SLURM_TEST_PLAN.md` results). This runbook takes the system
from the **old push model** (still running) to the **new pull model**.

## Current state at cutover time (2026-06-21)

- **Cluster:** 14 legacy push-model training jobs still RUNNING (8 on `rtx6000`,
  6 on `rtx_pro_6000`), e.g. `l2_8_init3`, `l2_4_init6`.
- **Prod DB** `/home/ashtomer/projects/ares/orchestrator/orchestrator.db`
  (botero mount: `~/slurm_mount/.../orchestrator.db`): **old schema**, 118 rows
  (77 PENDING / 27 COMPLETED / 14 RUNNING).
- **Migration preview** (run on a copy): old rows map to
  **83 AA_EVAL / 27 FINISHED / 8 PENDING**, ~83–85 claimable per partition.
- **Crontab:** old `cron_watchdog.sh` (`*/20`) **removed**; weekly sync kept;
  the hourly monitor line is **staged but commented**.

## ⚠️ The one hazard: do not double-run the 14 in-flight models

The first time new code opens the DB it migrates in place and **clears the owner
(`slurm_job_id`) on every non-terminal row** — including the 14 the legacy jobs
are still training. Once cleared they become claimable, so an active monitor
would submit controller pools that **re-claim and re-run those same models**.
Two processes writing the same `results/models/<name>/` dir can corrupt
checkpoints. So the 14 legacy jobs MUST be resolved before the monitor goes live.

Pick one:
- **(A) Drain** — let the 14 finish naturally, then cut over. Zero wasted work;
  can take days.
- **(B) Cancel + resume (recommended)** — `scancel` the 14; the migration carries
  `current_epoch → next_epoch`, so the controller **resumes each from its
  next_epoch**, losing at most the current in-progress epoch.

---

## Steps

### 0. Backup (always)
```bash
ssh slurm "cp /home/ashtomer/projects/ares/orchestrator/orchestrator.db \
              /home/ashtomer/projects/ares/orchestrator/orchestrator.db.pre_golive.$(date +%Y%m%d)"
```

### 1. Confirm cluster has the code
```bash
ssh slurm "cd /home/ashtomer/projects/ares && git checkout orchestrator-rebuild && git pull"
# HEAD should be the latest orchestrator-rebuild commit.
```

### 2. Resolve the 14 legacy jobs (choose A or B above)
```bash
# (B) cancel — they will resume from next_epoch after cutover:
ssh slurm "squeue -u ashtomer -h -o '%i %j' | grep -E 'l[0-9]|baseline|trades|gradnorm|v1' | awk '{print \$1}' | xargs -r scancel"
# verify only non-orchestrator jobs remain:
ssh slurm "squeue -u ashtomer -o '%i|%j|%T' -h"
```
(For (A), skip this and wait until `squeue` shows the 14 finished.)

### 3. Migrate + sanity-check the DB (deliberate, on botero)
```bash
cd ~/Documents/ares
PY=/home/tomer_a/miniconda3/envs/ares/bin/python
$PY -c "from orchestrator.db import OrchestratorDB; OrchestratorDB('$HOME/slurm_mount/projects/ares/orchestrator/orchestrator.db')"  # opening migrates in place
$PY -m orchestrator.cli status | tail -5    # expect ~83 AA_EVAL / 27 FINISHED / 8 PENDING, no stuck owners
```
Spot-check a couple of rows resume sanely:
```bash
$PY -m orchestrator.cli validate-model l2_8_init1   # status + next_epoch look right
```

### 4. Seed the first controller pools
Either run the monitor once by hand (it submits both pools), or submit directly.
```bash
$PY -m orchestrator.cli monitor          # one real pass: submits 1-200%6 + 1-200%8
ssh slurm "squeue -u ashtomer -n orch-controller -o '%i|%P|%T|%r' -h"
```
**Expect:** an array on each partition, `%6` (pro) / `%8` (rtx6000) running caps.
Watch a few rows move PENDING/AA_EVAL → TRAINING/… via `cli status`.

### 5. Enable the hourly monitor
```bash
crontab -e
# uncomment the staged line:
# 0 * * * * /home/tomer_a/miniconda3/envs/ares/bin/python -m orchestrator.monitor >> /home/tomer_a/Documents/ares/orchestrator/monitor.log 2>&1
crontab -l    # confirm it is active
tail -f ~/Documents/ares/orchestrator/monitor.log   # watch the first hourly tick
```

### 6. Steady-state verification (first 24–48h)
- `cli status` shows models advancing PENDING→TRAINING→AA_EVAL→PLOTTING→FINISHED.
- `monitor.log` tops up pools when remaining < 20 (afterany-chained, no double-queue).
- Finished `..._l2_2_init1` rows have `best_ckpt` + `best_score` (Test 4 in the
  test plan — first real confirmation of the deferred end-to-end checks 3/4/5).
- Any `FAILED` rows: deterministic reports in `orchestrator/recommendations/`;
  unknown signatures get a `codex` diagnosis (one per signature).

---

## Rollback
```bash
# stop the pull system:
crontab -e                                   # re-comment the monitor line
ssh slurm "scancel -u ashtomer -n orch-controller"
# restore the pre-cutover DB if needed:
ssh slurm "cp /home/ashtomer/projects/ares/orchestrator/orchestrator.db.pre_golive.<date> \
              /home/ashtomer/projects/ares/orchestrator/orchestrator.db"
```

## Notes
- **Email alerts are skipped** — no `mail` binary on botero. Failure diagnoses
  still land in `orchestrator/recommendations/`. Install `mailutils`/`s-nostmp`
  or point `ORCH_ALERT_EMAIL` at a working MTA if you want the emails.
- Deferred tests 3/4/5 (full train→AA→plot, best_checkpoint, resume) get their
  first real confirmation from step 6 — watch one model all the way through.
- The legacy push launchers (`golan-trainmodels.sbatch`, `stage_runner.sh`, the
  archived `orchestrator/legacy/*`) must not be submitted once live.
