# `aa_sweep` — daily AutoAttack full-sweep completion

Keeps every **finished** model on both clusters covered by the full AutoAttack grid:

```
linf | l2 | l1   ×   eps 1, 2, 4, 6, 8   ×   best | last | advbest      = 45 cells per model
```

Since `final_eval_aatype` defaults to `eps_norm` (commit `8fc067c4`), a finished training run only
gets AutoAttack at the single norm/eps it was *trained* on. This package finds the rest and runs it
on the BGU cluster (a DB-driven queue drained by two job arrays, `main` and `rtx6000`) and on
Botero's own 4090. eps 12 is deliberately **not** part of the grid;
existing eps-12 rows are left alone but never required.

## Flow

```
crontab ──▶ scripts/aa_sweep_daily.sh ──▶ python -m aa_sweep.submit
                                              │
   1. preflight   slurm sshfs mount + QNAP share up, local store present
   2. read DBs    frozen aircc_jobs (QNAP) + jobs.sqlite, read-only (immutable=1), 'finished'
   3. SPLIT       one batched ssh probe of the BGU dirs decides the lane per model
   4. CENSUS      each lane against its OWN machine's csvs: missing = grid − that machine's cells
   5. dedupe      squeue + the local queue: skip any (model, kind) a standalone job still holds
   6. dispatch    slurm lane → rows in the cluster queue (+ launch the arrays);
                  botero lane → rows in the local queue
```

## The Slurm lane: one queue, two arrays

```
21:30 cron ──▶ aa_sweep.submit ──ssh──▶ cluster_queue feed --launch     (login node, python3)
                                          │  upsert units into aa_sweep/aa_queue.sqlite
                                          │  release rows whose owning task is dead
                                          └─ per lane: pending work AND no pending task of its
                                             array → sbatch --dependency=singleton <lane>.sbatch
aaq-main     sbatches/aa_sweep_queue_main.sbatch      main,    --array=1-200%8
aaq-rtx6000  sbatches/aa_sweep_queue_rtx6000.sbatch   rtx6000, --array=1-200%8, --qos=golan-neuro
   each task (scripts/aa_queue_task.sh): GPU check → claim ONE unit → engine → finish
```

* **One task = one (model, kind) unit.** `claim` runs on the stock `python3` before conda loads.
* **The CSV decides, not the exit code.** `finish` re-censuses the model dir with the same
  `census.kind_status` the planner uses: nothing missing → `finished`; otherwise the attempt counts
  and the unit goes back to `pending`, parking as `failed` after `QUEUE_MAX_ATTEMPTS` (3).
* **Nothing pending → nothing running.** A task that finds nothing to claim cancels its array's
  remaining PENDING tasks and exits; the feed submits no array while nothing is claimable.
* **`--dependency=singleton`** makes a new array of a lane wait for that lane's previous array to
  end, so a lane never exceeds its `%8`. A lane is only resubmitted when its array has no task still
  waiting to start.
* **Time limit / scancel** (`--signal=B:TERM@120`): the task kills the engine first (never two
  writers on one CSV), then `release`s the unit without counting an attempt. The engine flushes
  after every cell, so the next claimant resumes exactly there. A task that dies without the trap
  firing is caught by `requeue_dead` (squeue → sacct, fails open, 5-min grace) at the next claim
  or feed.
* **Standalone jobs still win.** Units a hand-submitted eval (or a pre-queue `aaswp_*` sbatch)
  still holds are not fed until that job leaves squeue.

### Operating it (on the cluster, from the repo root)

```bash
python3 -m aa_sweep.cluster_queue status [--all]     # counts, running, failed, next up
python3 -m aa_sweep.cluster_queue reset <id>         # re-arm a failed unit
python3 -m aa_sweep.cluster_queue drop <id>          # delete a unit
echo '[]' | python3 -m aa_sweep.cluster_queue feed --launch   # launch due arrays, feed nothing
scancel -n aaq-rtx6000                               # stop a lane (rows are released on next claim/feed)
```

From Botero: `ssh slurm "cd /home/ashtomer/projects/ares && python3 -m aa_sweep.cluster_queue status"`.
Per-task logs: `outs/aa_sweep/<array>_<task>.out`.

### Cutover from per-unit sbatch (2026-09-28)

The old driver submitted one `aaswp_<model>_<kind>` sbatch per unit on `main`. At cutover the
PENDING ones were cancelled and re-fed into the queue; RUNNING ones were left to finish (the feed
skips their units while they are in squeue, and closes them once their CSV is complete):

```bash
ssh slurm "squeue -u ashtomer -h -t PENDING -o '%i %j' | awk '\$2 ~ /^aaswp_/ {print \$1}' | xargs -r scancel"
AA_SWEEP_ARGS=--dry-run aa_sweep/scripts/aa_sweep_daily.sh     # preview the feed + launch
aa_sweep/scripts/aa_sweep_daily.sh                             # feed + launch both arrays
```

## Two lanes, no transfers

The package used to be one cross-machine system: it rsync'd ~1.4 GB checkpoints from the archive
onto the BGU cluster, pushed merged CSV text over ssh, and let the local lane `scancel` Slurm jobs
to steal them. All of that is gone. **Each machine evaluates the copy of the model it already
holds**, and writes its results back beside that copy:

| | model source | results land in |
|---|---|---|
| **Slurm** | `/home/ashtomer/projects/ares/results/models/<name>/` — the cluster's own disk | the same dir |
| **Botero** | `/mnt/data4t/models/<arch>/<name>/` — the `model_store` curated tree | the same dir |

Propagating results between the machines is **not this package's job**. The weekly rsync carries
whatever is in a model dir, sweep CSVs included; nothing here copies a model, and nothing here
pushes a result.

The only things that still cross the network: one batched read-only ssh probe of the cluster's
model dirs, `squeue`, and one ssh `feed` into the cluster queue.

### Which lane owns a model

`plan.build_plan` decides once, and the split is **disjoint by construction**:

* **Slurm lane** — the ssh probe found a directory for the model on the cluster, whichever DB
  reported it finished. 102 of the 127 AIRCC-finished models are here: the old staging step already
  copied them over, so the cluster owns them now and this machine never touches them again.
* **Botero lane** — finished on AIRCC, *and* the cluster has no directory for it. 22 models at last
  count. Three more (`convnext_base_{l2_cont4to6,linf_cont4to6,linf_cont4to8}_init0`) never wrote a
  checkpoint at all and are simply dropped.
* Anything finished only by sjm with no cluster dir is dropped too — the cluster cannot attack what
  it does not have, and it is not this machine's to run.

Disjointness is what makes the per-lane census correct: see *Things that are easy to get wrong*.

### The AIRCC side is a list of names, nothing more

This machine does not contact the AIRCC cluster — no `ssh aircc`, no `~/aircc_mount`. All that
survives of the allocation here is the finished-model list in the frozen
`/mnt/botero/aircc_archive/aircc_jobs_final_latest.sqlite` (127 of 323 rows, byte-for-byte the set
the live DB last reported). The checkpoints come from the local store, not the archive.

## The Botero lane

Botero's RTX 4090 is the same 24 GB class the sbatch demands via `--constraint`, so it is a second
lane rather than a helper for the first:

```
21:30 cron ──▶ aa_sweep.submit ──▶ botero.topup()      keep 7 units queued locally
                                        │
*/10 cron  ──▶ aa_sweep_botero_runner.sh ──▶ botero_runner.tick()
                                        │
   flock (held for the whole job)  ──▶  one job at a time, never two on one card
   GPU gate (any foreign CUDA proc) ──▶  defer; the workstation's user always wins
   claim oldest queued  ──▶  autoattack_array_eval.py --model-dir <local store dir>
```

**Depth 7, concurrency 1.** `BOTERO_SLOTS` is a *backlog*, not a parallelism level. A full 14-cell
checkpoint is days of work on one card; the backlog exists so the lane never idles waiting for the
next nightly top-up.

**Fullest-first.** Top-up takes the checkpoints with the most missing cells: finishing one
checkpoint completely is worth more than a cell each on three of them.

**Model resolution is by directory basename.** The store nests models one level under an
architecture dir (`convnext_base/convnext_base_baseline_init0`) while the DB names are flat (AIRCC)
or nested under a *different* prefix (sjm's `vit_b_cvst/l1_1_init1`), so `botero.store_index()`
indexes the tree by basename. Verified safe: 331 model dirs, 331 distinct basenames, no collisions.
A dir qualifies only if it holds the checkpoint *and* `autoattack_sweep_selection.json` — without
the selection the run would attack a different 1024 images and the rows would not be comparable.

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
python -m aa_sweep.submit                    # feed the cluster queue + top up the local queue
python -m aa_sweep.submit --model convnext_base_linftrades_2_init0   # one model (debugging)
python -m aa_sweep.submit --limit 3          # feed at most 3 runnable units (debugging)
```

Install the cron (see the header of `scripts/aa_sweep_daily.sh`):

```cron
30 21 * * * /home/tomer_a/Documents/ares/aa_sweep/scripts/aa_sweep_daily.sh >> /home/tomer_a/Documents/ares/aa_sweep/logs/aa_sweep.log 2>&1
```

Quiet unless something breaks; emails via `aircc.aircc_job_manager.notify.make_emailer` on a
missing filesystem, an unreadable DB, or a failed queue feed / array sbatch.

## Files

| File | Role |
|---|---|
| `config.py` | Paths, ssh hosts, the grid, checkpoint↔CSV mapping. All env-overridable. |
| `census.py` | Pure: one machine's CSV text → which `(norm, eps)` cells its checkpoint still needs. |
| `plan.py` | Both DBs + one batched ssh probe → the lane split, one `ModelWork` per model. |
| `submit.py` | Entrypoint: preflight → plan → dedupe → cluster-queue feed + local top-up. |
| `cluster_queue.py` | The cluster queue (runs on the cluster, stdlib only): feed/claim/finish/release, array launch, `status`. |
| `scripts/aa_queue_task.sh` | Body of one array task; sourced by both `sbatches/aa_sweep_queue_*.sbatch`. |
| `scripts/aa_sweep_daily.sh` | Cron wrapper: `flock`, notify-on-crash. |
| `botero.py` | The local lane: queue DB, store resolution, top-up, `status`/`enqueue` CLI. |
| `botero_runner.py` | The local worker: GPU gate, claim, run the engine. |
| `scripts/aa_sweep_botero_runner.sh` | Cron wrapper for the worker: `flock`, hold file, quiet ticks. |

Cluster side: each array task runs `data_analysis/autoattack_array_eval.py --model-dir …
--checkpoint-kinds <one kind>`. `sbatches/aa_sweep_completion.sbatch` is the same command as a
standalone job, kept for one-off manual runs (its `aaswp_*` name keeps the feed off that unit).

## Things that are easy to get wrong

- **Nothing is ever recomputed.** `--force` is never passed. The engine diffs the CSV's existing
  `(norm, eps)` rows against the grid and attacks only the difference, so an eps_norm row already on
  disk is reused as-is — that is the entire point.
- **1024 images, always.** The sbatch uses `--batch-size 32 --num-batches 32` (= 1024) to match the
  older `128 × 8` sweeps, and `autoattack_sweep_selection.json` is reused, so new rows are attacking
  the *same* images as the old ones and are directly comparable.
- **One unit per kind, never per norm.** The three kinds write three different CSVs and cannot race.
  Splitting further (per norm) would put three jobs on one CSV and would need a lock.
- **CSV rows key off the directory basename.** sjm names are nested (`vit_b_cvst/linf_1_init1`) but
  `model_name` in the CSV is `linf_1_init1`. Job names flatten `/` to `__` so they survive
  `squeue -o %j`.
- **Dedupe decodes `aaswp_*` names, it does not pattern-match them.** Every such name is ours, so
  only this unit's own names may block; the loose dir-name token is for *foreign* jobs only.
  Matching that token against our own names cost swin_b and vit_b_cvst 16 models: their dir name
  (`linf_cont4to6_init1`) is a `_`-delimited suffix of an unrelated flat
  `convnext_base_linf_cont4to6_init1`, whose job had been PENDING behind `QOSMaxGRESPerUser` for
  weeks — so "skip it, next night will pick it up" never came due.
- **A nested job has two names, and both must block.** `aa_sweep_completion.sbatch` renames a job to
  `aaswp_$(basename AA_MODEL_DIR)_<kind>` when it *starts*, so `vit_b_cvst/l2_cont4to6_init1` is
  queued as `aaswp_vit_b_cvst__l2_cont4to6_init1_best` and runs as `aaswp_l2_cont4to6_init1_best`.
  `config.own_job_names` returns both; checking only the submitted form would put a second job on a
  CSV a running one already owns.
- **Each lane censuses only its own machine's CSVs.** Deliberately *not* a union. What ultimately
  decides which cells get attacked is the engine on that machine diffing its own CSV against the
  grid, so counting a row that lives only on the other machine would make the planner skip a cell
  that then never gets computed. This is correct only because the lanes own **disjoint** model sets
  — loosen that split and this is the assumption that breaks.
- **The Botero queue is part of the dedupe set.** `live_job_names()` unions the local queue's rows
  in under the same `aaswp_<model>_<kind>` names. Belt-and-braces given the disjoint split, but it
  costs one local sqlite read and guarantees one unit never lands in two lanes.
- **Nothing here moves a model or a result.** No rsync, no scp, no `scancel`. The only network calls
  are the read-only ssh probe, `squeue` and `sbatch`. If you find yourself adding a copy step, the
  answer is almost certainly the weekly rsync instead.
- **Runtime is hours per cell.** ~25–40 h per checkpoint on a 3090/4090. The engine flushes the CSV
  after every setting, so a job that hits the 7-day limit loses at most one cell and the next day's
  cron picks up exactly where it stopped.
