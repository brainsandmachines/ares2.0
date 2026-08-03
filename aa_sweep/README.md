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
   4. stage       rsync AIRCC → BGU, --ignore-existing, ONLY the gapped kinds' checkpoints
   5. dedupe      squeue: skip any (model, kind) already pending/running
   6. submit      one sbatch per (model, kind) → sbatches/aa_sweep_completion.sbatch
```

## Usage

```bash
python -m aa_sweep.submit --dry-run          # print the plan, touch nothing
python -m aa_sweep.submit                    # stage + submit
python -m aa_sweep.submit --model convnext_base_linftrades_2_init0   # one model (debugging)
python -m aa_sweep.submit --limit 3          # cap submissions (debugging)
```

Install the cron (see the header of `scripts/aa_sweep_daily.sh`):

```cron
30 7 * * * /home/tomer_a/Documents/ares/aa_sweep/scripts/aa_sweep_daily.sh >> /home/tomer_a/Documents/ares/aa_sweep/logs/aa_sweep.log 2>&1
```

07:30 follows the 03:00 AIRCC backup and the 06:00 sjm status check. Quiet unless something breaks;
emails via `aircc.aircc_job_manager.notify.make_emailer` on mounts down, unreadable DB, or a failed
rsync/sbatch.

## Files

| File | Role |
|---|---|
| `config.py` | Paths, ssh hosts, the grid, checkpoint↔CSV mapping. All env-overridable. |
| `census.py` | Pure: CSV text → which `(norm, eps)` cells a checkpoint still needs. |
| `plan.py` | Both DBs + one batched ssh probe → one `ModelWork` per finished model. |
| `stage.py` | The `--ignore-existing` rsync and the AIRCC-only CSV row merge. |
| `submit.py` | Entrypoint: preflight → plan → stage → dedupe → sbatch. |
| `scripts/aa_sweep_daily.sh` | Cron wrapper: `flock`, notify-on-crash. |

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
- **Runtime is hours per cell.** ~25–40 h per checkpoint on a 3090/4090. The engine flushes the CSV
  after every setting, so a job that hits the 7-day limit loses at most one cell and the next day's
  cron picks up exactly where it stopped.
