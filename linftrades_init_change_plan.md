# Plan: linftrades cont4to8 init change

**Status: DONE (2026-10-22), except steps 4–5, which the weekly crons finish on their own.**
Run on the user's go-ahead before the watcher fired. At that time init2 was `finished` and then
re-opened by `reopen_eval` for the rest of its AA grid. init3 was still in final_eval (advbest cell
running), and its DB row had no `best_score` yet. Gate 3 passed on the AA linf@8 sweep CSVs:

| new run | best | last | advbest | old |
|---|---|---|---|---|
| init2 | 8.01 (ep0) | **10.64** (ep39) | 10.64 (ep39) | init0 1.86 |
| init3 | 9.86 (ep0) | **17.58** (ep39) | pending (ep38) | init1 4.49 |

- Step 1: done. Both dirs are in `results/models_failed/…__collapsed_20261007`.
- Steps 2–3: done and committed. A zoo dry-run reports both overrides as `override-unused` until
  init2/init3 reach `/mnt/data4t/models` (Sun route 1 → Mon route 2). Until then the live zoo
  still points init0/init1 at the old collapsed checkpoints.
- Step 4: pending. After Monday's sync, check that `models_for_experiments/manifest.csv` maps
  init0/init1 to the init2/init3 checkpoints.
- Step 5: pending. Approve the Monday HF proposal by number.
- Step 6: done. The crontab entry is removed (backup in `slurm_job_manager/logs/crontab_backup_20261022.txt`)
  and the watcher script is deleted.

## Background

The completion runs (see `completion_runs_plan.md`, prio 301/302) retrain the two collapsed
convnext_base linf TRADES 4→8 continuations from the same parents:

| new run | starts from | replaces | old AA linf@8 |
|---|---|---|---|
| `convnext_base_linftrades_cont4to8_init2` | `convnext_base_linftrades_4_init0` → `model_best_adv` | `convnext_base_linftrades_cont4to8_init0` | 1.86 (last) |
| `convnext_base_linftrades_cont4to8_init3` | `convnext_base_linftrades_4_init1` → `last` | `convnext_base_linftrades_cont4to8_init1` | 4.49 (last) |

On 2026-10-19 both new runs were still training (epoch 28/40 and 27/40, no AA yet). Their PGD-val
adv top-1 (~46 / ~45) matched the old init0/1, which still collapsed under AutoAttack. So PGD-val
proves nothing here; only the AA score decides.

## Gate: check before doing anything

1. Both rows are `finished` in the sjm DB, with a `best_score`.
2. Both cluster dirs have `autoattack_eps_norm_scores.json` and the `autoattack_sweep_results*.csv`.
3. **The new AA at linf@8 clearly beats the old one** (init2 > 1.86, init3 > 4.49, and ideally
   ≥ 10 so the HF proposal does not flag it). If either still looks collapsed, stop and tell the
   user. Do not run the steps below.

## Steps (after the user confirms)

1. **Move the failed init0/1 to `models_failed` on the cluster, together with their CSVs** (the
   whole dir: checkpoints, the 3 `autoattack_sweep_results*.csv`, the scores json, the png, the logs):
   ```bash
   ssh slurm 'cd /home/ashtomer/projects/ares/results && \
     mv models/convnext_base_linftrades_cont4to8_init0 models_failed/convnext_base_linftrades_cont4to8_init0__collapsed_20261007 && \
     mv models/convnext_base_linftrades_cont4to8_init1 models_failed/convnext_base_linftrades_cont4to8_init1__collapsed_20261007'
   ```
   The Sunday backup (`backup_slurm_models.sh`, leg `models_failed`) copies them to the QNAP.
   Both are already listed in `aa_sweep/sweep_excludes.csv`, so nothing re-sweeps them.
   Leave the Botero store dirs (`/mnt/data4t/models/convnext_base/…_init{0,1}`) alone. The QNAP
   still holds them under `models/`, and it is read-only, so the Monday backfill would just copy
   them back. Step 2 is what keeps them out of the zoo.
2. **Repoint the zoo names.** Add two rows to `model_store/blessing_overrides.csv` (`publish_as`,
   the same method as the `pgd5` reruns):
   - `convnext_base_linftrades_cont4to8_init2` → `publish_as` `convnext_base_linftrades_cont4to8_init0`
   - `convnext_base_linftrades_cont4to8_init3` → `publish_as` `convnext_base_linftrades_cont4to8_init1`

   Each row needs a `reason` citing the old and new AA scores. `epsilon_bounded_contstim` keeps
   loading the same init0/init1 names.
3. **Remove the init0/init1 rows from `model_store/hf_excludes.csv`.** It matches on the
   *published* name, so after step 2 those rows would keep the good models off the Hub.
4. **Rebuild.** After the cluster row is `finished`, the Sunday route 1 + Monday route 2 do it on
   their own. To do it sooner, run `model_store/scripts/ms_run.sh backfill-apply` then `zoo-apply`
   (dry-run each first). Check that `models_for_experiments/manifest.csv` maps init0/init1 to the
   init2/init3 checkpoints. Commit steps 2–3.
5. **HF.** The next Monday proposal mail should list `convnext_base_linftrades_cont4to8_init0` and
   `init1` as `new`, with `metadata.json` provenance pointing at the init2/init3 dirs. The user
   approves it by number ("approved mail #N").
6. **Clean up the watcher.** Remove the `watch_linftrades_init_change` crontab entry. Then delete
   `slurm_job_manager/watch_linftrades_init_change.py` and mark this file DONE.

## The watcher

```bash
python -m slurm_job_manager.watch_linftrades_init_change --dry-run   # preview, send and save nothing
```

- Cron: `0 8,20 * * *`; log in `slurm_job_manager/logs/watch_linftrades_init_change.log`.
- State: `slurm_job_manager/logs/watch_linftrades_init_change.json`.
- It mails once (urgent, not held for the digest) when both runs are ready. It also mails once
  if either row ends `failed`. A dropped `~/slurm_mount` is logged and never counts as finished.
