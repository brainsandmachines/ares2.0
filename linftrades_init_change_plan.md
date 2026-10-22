# Plan: linftrades cont4to8 init change

**Status: DONE (2026-10-22).** Both reruns are in the zoo and on the Hub under the init0/init1 names.
Run on the user's go-ahead before the watcher fired. AA linf@8 (1024 images):

| new run | best | last | advbest | old | published as |
|---|---|---|---|---|---|
| init2 | 8.01 (ep0) | **10.64** (ep39) | 10.64 (ep39) | init0 1.86 | init0 (HF #4) |
| init3 | 9.86 (ep0) | **17.58** (ep39) | 17.29 (ep38) | init1 4.49 | init1 (HF #3) |

- Step 1: done. Both old dirs are in `results/models_failed/…__collapsed_20261007`, and copied to
  the QNAP's `models_failed` by the hand-run route 1.
- Steps 2–3: done. `publish_as` overrides were added and the init0/init1 `hf_excludes` rows removed.
- Step 4: done 2026-10-22, outside the weekly schedule.
  - init3: route 1 run by hand (separate pacing file
    `logs/.backup.attempted.manual_20261022`, so Sunday's run is unaffected), then
    `ms_weekly_sync.sh`.
  - init2: it was in its reopen_eval AA-grid rerun (`running`), so route 1 skipped it. I rsynced
    its dir from the cluster straight into `/mnt/data4t/models/convnext_base/` (route 1's
    filters; `last.pth.tar` sha256 a8907a02… matches the cluster). Then I pinned
    `last.pth.tar` + score 10.64 on its override row, because the zoo's trained gate only
    sees the QNAP and `finished` rows. After `zoo-apply` + `--check` (204 rows, 0 problems),
    `models_for_experiments` maps init0 → `…_init2/last.pth.tar` and init1 → `…_init3/last.pth.tar`.
  - Later weekly runs: route 1 copies init2 to the QNAP once its row is finished. The
    checkpoints are byte-identical, so backfill only refreshes its AA CSVs, and HF sees no change.
- Step 5: done. HF #2 (it held the collapsed init0) was superseded. #3 uploaded init1 and #4
  uploaded init0; each passed `verify_hf --forward` (8/8 top-1 agree).
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
