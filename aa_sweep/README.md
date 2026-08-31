# `aa_sweep` — daily AutoAttack full-sweep completion

Keeps every **finished** model on both clusters covered by the full AutoAttack grid:

```
linf | l2 | l1   ×   eps 1, 2, 4, 6, 8   ×   best | last | advbest      = 45 cells per model
```

Since `final_eval_aatype` defaults to `eps_norm` (commit `8fc067c4`), a finished training run only
gets AutoAttack at the single norm/eps it was *trained* on. This package finds the rest and runs it
on the BGU cluster's shared `main` partition. eps 12 is deliberately **not** part of the grid;
existing eps-12 rows are left alone but never required.

## Flow

```
crontab ──▶ scripts/aa_sweep_daily.sh ──▶ python -m aa_sweep.submit
                                              │
   1. preflight   both sshfs mounts up, ssh slurm reachable
   2. read DBs    aircc_jobs.sqlite + jobs.sqlite, read-only (immutable=1), status='finished'
   3. CENSUS      one batched ssh probe of the BGU dirs + local read of the AIRCC mount
                  → missing(kind) = grid − (AIRCC cells ∪ BGU cells)
                  → models with nothing missing stop here: no rsync, no job, no bytes
   4. stage       rsync → BGU, --ignore-existing, ONLY the gapped kinds' checkpoints.
                  Source is the local mirror when verified fresh, else the AIRCC mount.
   5. dedupe      squeue: skip any (model, kind) already pending/running
   6. submit      one sbatch per (model, kind) → sbatches/aa_sweep_completion.sbatch
```

### Staging source

Reading the AIRCC sshfs mount measures ~3.4 MB/s, so a 2.8 GB model takes ~14 minutes. The 03:00
backup cron already archives that tree to local disk at `/mnt/data4t/aircc_archive/models`,
so `mirror.py` prefers the mirror — but only when it is *provably* a faithful copy:

1. **Global** — the latest `backup.log` block validated, via
   `aircc.aircc_job_manager.daily_monitor.check_backup_log` (one definition of "backup succeeded").
2. **Per file** — every file about to be staged (checkpoints *and* the CSVs/selection json that
   ride along) matches the AIRCC source on both size and mtime. `rsync -rt` preserves mtime, so a
   mismatch means stale or mid-flight.

Either gate failing falls back to the AIRCC mount for that model, and the reason is logged. A model
that finished after the last backup is simply not mirrored yet and takes the same fallback. Set
`AA_SWEEP_USE_MIRROR=0` to force the mount.

The cron runs at **21:30**, 18.5 h after the 03:00 backup starts. The backup itself now runs only
every 3 days and its full pass takes ~16.5 h (03:00 -> ~19:35 on 2026-08-08); a 19:30 cron raced it
by five minutes and lost the mirror for that night, so the slot was moved back two hours.

## The Botero lane

The BGU queue is chronically jammed behind `QOSMaxGRESPerUser` — on 2026-08-30, 163 of 170 `aaswp`
jobs were PENDING and 154 of those still needed 14 of their 15 cells. Botero's own RTX 4090 is the
same 24 GB class the sbatch demands via `--constraint`, so it is a second lane:

```
21:30 cron ──▶ aa_sweep.submit ──▶ botero.topup()      keep 5 units queued locally
                                        │
*/10 cron  ──▶ aa_sweep_botero_runner.sh ──▶ botero_runner.tick()
                                        │
   flock (held for the whole job)  ──▶  one job at a time, never two on one card
   GPU gate (any foreign CUDA proc) ──▶  defer; the workstation's user always wins
   claim oldest queued  ──▶  autoattack_array_eval.py --model-dir <local archive>
```

**Depth 5, concurrency 1.** `BOTERO_SLOTS` is a *backlog*, not a parallelism level. A full 14-cell
checkpoint is days of work on one card; the backlog exists so the lane never idles waiting for the
next nightly top-up.

**Moving beats adding.** Top-up prefers to take work that is already queued on Slurm — `scancel`
there, enqueue here — starting from the **highest job id**, i.e. the jobs Slurm would pick up last.
Only when there is nothing left to move does it enqueue work that was never queued anywhere. A unit
still pending on Slurm is never enqueued locally, so nothing can run in both lanes.

**Models come from the local archives**, searched in order: `/mnt/data4t/slurm_archive/models`,
`/mnt/data4t/aircc_archive/models`, then the two `/mnt/botero` (QNAP) twins. A directory qualifies
only if it holds both the checkpoint *and* `autoattack_sweep_selection.json` — without the selection
the run would attack a different 1024 images and the rows would not be comparable. The two archives
are not cleanly split by originating cluster, so all four roots are searched.

**Results stay on Botero.** Rows are written into the archive dir they were read from and echoed
(CSV + comparison PNG only) onto the QNAP twin; nothing is pushed back to either cluster. That is
why `census.kind_status` takes a third source — see *Things that are easy to get wrong* below.

### Operating it

```bash
python -m aa_sweep.botero status                  # the queue
python -m aa_sweep.botero status --all            # including finished/failed
python -m aa_sweep.botero enqueue <model> <kind>  # queue one by hand
python -m aa_sweep.botero reset <id>              # a failed row back into the queue
python -m aa_sweep.botero drop <id>               # delete a row

python -m aa_sweep.submit --botero-topup-only     # top up now, submit no sbatch
python -m aa_sweep.submit --no-botero             # cluster only
aa_sweep/scripts/aa_sweep_botero_runner.sh        # one tick by hand
AA_BOTERO_ARGS=--check-gpu aa_sweep/scripts/aa_sweep_botero_runner.sh
date -Is -d 'tomorrow 06:00' > aa_sweep/logs/.botero_hold   # pause the lane
```

Per-job logs are `aa_sweep/logs/botero/<model>__<kind>.log`; the cron's own log is
`aa_sweep/logs/botero_runner.log`, quiet on idle ticks.

## Usage

```bash
python -m aa_sweep.submit --dry-run          # print the plan, touch nothing
python -m aa_sweep.submit                    # stage + submit
python -m aa_sweep.submit --model convnext_base_linftrades_2_init0   # one model (debugging)
python -m aa_sweep.submit --limit 3          # cap submissions (debugging)
```

Install the cron (see the header of `scripts/aa_sweep_daily.sh`):

```cron
30 21 * * * /home/tomer_a/Documents/ares/aa_sweep/scripts/aa_sweep_daily.sh >> /home/tomer_a/Documents/ares/aa_sweep/logs/aa_sweep.log 2>&1
```

Quiet unless something breaks; emails via `aircc.aircc_job_manager.notify.make_emailer` on mounts
down, unreadable DB, or a failed rsync/sbatch.

## Files

| File | Role |
|---|---|
| `config.py` | Paths, ssh hosts, the grid, checkpoint↔CSV mapping. All env-overridable. |
| `census.py` | Pure: CSV text → which `(norm, eps)` cells a checkpoint still needs. |
| `plan.py` | Both DBs + one batched ssh probe → one `ModelWork` per finished model. |
| `stage.py` | The `--ignore-existing` rsync and the AIRCC-only CSV row merge. |
| `submit.py` | Entrypoint: preflight → plan → stage → dedupe → sbatch. |
| `scripts/aa_sweep_daily.sh` | Cron wrapper: `flock`, notify-on-crash. |
| `botero.py` | The local lane: queue DB, archive-dir resolution, top-up, `status`/`enqueue` CLI. |
| `botero_runner.py` | The local worker: GPU gate, claim, run the engine, echo to the QNAP. |
| `scripts/aa_sweep_botero_runner.sh` | Cron wrapper for the worker: `flock`, hold file, quiet ticks. |

Cluster side: `sbatches/aa_sweep_completion.sbatch` runs
`data_analysis/autoattack_array_eval.py --model-dir … --checkpoint-kinds <one kind>`.

## Things that are easy to get wrong

- **Nothing is ever recomputed.** `--force` is never passed. The engine diffs the CSV's existing
  `(norm, eps)` rows against the grid and attacks only the difference, so an eps_norm row already on
  disk is reused as-is — that is the entire point.
- **1024 images, always.** The sbatch uses `--batch-size 32 --num-batches 32` (= 1024) to match the
  older `128 × 8` sweeps, and `autoattack_sweep_selection.json` is reused, so new rows are attacking
  the *same* images as the old ones and are directly comparable.
- **One job per kind, never per norm.** The three kinds write three different CSVs and cannot race.
  Splitting further (per norm) would put three jobs on one CSV and would need a lock.
- **`--ignore-existing` is load-bearing.** Ten AIRCC model names already exist as BGU directories,
  and five of those are real BGU-trained `*pgd5*` runs whose AIRCC counterpart is a near-empty husk.
  The rsync can only add files; size mismatches are logged as `CONFLICT` and left alone.
- **CSV rows key off the directory basename.** sjm names are nested (`vit_b_cvst/linf_1_init1`) but
  `model_name` in the CSV is `linf_1_init1`. Job names flatten `/` to `__` so they survive
  `squeue -o %j`.
- **Botero results are invisible to the cluster.** They are never pushed back, so `kind_status`
  reads the local archive as a *third* census source next to the BGU probe and the AIRCC mount.
  Drop that and the nightly run resubmits to Slurm every cell Botero has already computed.
- **The Botero queue is part of the dedupe set.** `live_job_names()` unions the local queue's rows
  in under the same `aaswp_<model>_<kind>` names, so a unit this machine owns is never also sent to
  the cluster. One unit, one lane.
- **Runtime is hours per cell.** ~25–40 h per checkpoint on a 3090/4090. The engine flushes the CSV
  after every setting, so a job that hits the 7-day limit loses at most one cell and the next day's
  cron picks up exactly where it stopped.
