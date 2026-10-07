# Collapsed-run rerun plan (catastrophic overfitting / gradient masking)

*Written 2026-10-07; revised the same day. **This is a plan.** Only one action has been taken: the
AutoAttack sweep exclusion (§0). Every other step is still to do and needs an explicit go-ahead.*

**Revision 2026-10-07: one approach for every collapsed model.** Each collapsed run keeps its
original directory and gets a **new continuation run**. The new run starts from the original's
DB-best checkpoint and trains **40 more epochs at the same ε with 7 PGD steps**. This is the same
recipe as the existing cont4to6/cont4to8 runs, except the target ε equals the source ε. The only
models not using this approach are `convnext_base_linftrades_cont4to8_init{0,1}` (Task 1, unchanged).
`vit_b_cvst/l2trades_4_init1` joins the list.

**Revision 2 (2026-10-07): no PGD step warmup in continuation runs** (§2, "Code change before any
continuation run"). A continuation restarts its epoch counter at 0, so the from-scratch step warmup
`attack_step * min(epoch, 5) / 5` trains a robust model on clean images for a whole epoch. That is
why cont4to8 collapsed. The fix lands in code before Task 1 and recipe A run.

## 0. Done 2026-10-07: these models are out of the AA sweep

- `aa_sweep/sweep_excludes.csv` lists the 12 original models in §1 (11, plus `vit_b_cvst/l2trades_4_init1`). `aa_sweep.submit` drops them before
  planning, so the nightly cron never feeds them to either lane. It logs one `excluded <name>: <reason>`
  line per model.
- The cluster queue held 13 of their units: pending ids 132-134, 147-152 and 202-204, plus running
  id 75. All 13 were dropped with `cluster_queue drop`. Array task `aaq-main` 22156585_6, which held
  unit 75, was left to finish its cells.
- `vit_b_cvst/l2trades_4_init1` was added to the list later the same day. Its 3 pending units
  (ids 184-186) were dropped too.
- The new continuation runs (§3) have new names, so the sweep picks them up automatically once
  sjm marks them `finished`.
- **Undo:** delete a model's line from the CSV. The next 21:30 feed re-inserts its units.

## 1. What the data says

- **The in-training PGD validation cannot detect this failure.** `validate.py` attacks with the
  training `attack_it`, which is 3 by default. A masked model fools it: `swin_b/linftrades_4_init1`
  scores PGD-val 80.1 against clean 80.3, and AA 0.00. `model_best` and `model_best_adv` are chosen by
  that metric, so the DB-best checkpoints below are mostly collapsed epochs. The new runs validate
  with 7 steps, which is better but still weak, so trust only the AA sweep.
- **Collapsed models have higher clean accuracy than the healthy reference.** The model gives up
  robustness and gets clean accuracy back. For example:
  - swin linftrades_4: 79.1 clean, against 76.6 for swin linf_4 Madry.
  - vit linftrades_4: 73.7, against 70.3.

  dvd_b_l2_4_init1 is the exception. It diverged instead: clean fell from 66 to 54 during training.
- **Most of these used the default `attack_it: 3`.** dvd_b_l2_4_init1 and dvd_b_l1_4_init0 used 5.
  Madry training also hard-codes `random_start=False` (`ares/utils/train_loop.py:79`).
- **Several are repeat failures** (see `results/models_failed/` and
  `/mnt/botero/aircc_archive/models_failed/`):
  - vit linftrades_4: attempt 3
  - dvd_b_l2_2_init0: attempt 3
  - dvd_b_l1_4_init0: attempt 3
  - dvd_b_linf_2_init0: attempt 2

### Starting points: the DB-best checkpoint of each original

| original model (norm, ε) | DB-best file | epoch | clean | AA there | healthy reference (clean / AA) |
|---|---|---|---|---|---|
| swin_b/linftrades_4_init1 (linf 4) | model_best | 146 | 79.1 | 0.00 | swin_b/linf_4_init1: 76.6 / 50.5 |
| swin_b/linftrades_2_init1 (linf 2) | model_best | 125 | 80.9 | 0.20 | swin_b/linf_2_init1: 77.7 / 62.3 |
| swin_b/l2trades_4_init1 (l2 4) | model_best | 82 | 77.0 | 0.88 | swin_b/l2_4_init1: 70.5 / 38.2 |
| vit_b_cvst/linftrades_4_init1 (linf 4) | model_best_adv | 142 | 74.3 | 0.39 | vit_b_cvst/linf_4_init1: 70.3 / 40.8 |
| vit_b_cvst/l2trades_4_init1 (l2 4) | model_best | 113 | 72.2 | 12.01 | vit_b_cvst/l2trades_6: 17.4 at ε6 (ε4 should be higher) |
| convnext_base_dvd_b_linf_2_init0 (linf 2) | last | 199 | 72.9 ‡ | 0.10 | dvd_b_linf_2_init1: 68.8 / 47.9 † |
| convnext_base_dvd_b_l2_2_init0 (l2 2) | last | 199 | 68.3 | 0.20 | dvd_b_l2_2_init1: 70.8 / 50.6 |
| convnext_base_dvd_b_l2_4_init1 (l2 4) | last | 199 | 58.1 | 8.79 | dvd_b_l2_4_init0 (last): 57.2 / 26.3 |
| convnext_base_dvd_b_l1_4_init0 (l1 4) | model_best | 199 | 69.7 | 4.98 | dvd_b_l1_4_init1: 74.0 / 13.8 |
| convnext_base_dvd_b_l2_cont4to6_init1_contepoch (l2 6) | model_best | 234 | 66.9 | 0.10 | ..._init0_contepoch: 59.9 / 14.8 |

The AIRCC DB still points the convnext rows at the dead `/shared/cycle2_bgu_golan_prj/...` paths. The
same filenames exist in the cluster copies, and for these 10 models each cluster copy's AA matches
its DB score.

† The cluster copy of `dvd_b_linf_2_init1` is a stale, collapsed earlier attempt; the good one lives
on the QNAP. A separate audit (`aircc_copy_audit.md`) is checking every AIRCC model for the same
problem. Read its result before relying on any convnext cluster copy.

‡ Corrected 2026-10-07 (`aircc_copy_audit.md`, step 5). The row first said clean 78.6, which belongs to
the cluster copy of the *failed first attempt*. The DB-blessed run is attempt 2 on the QNAP, whose
`last` (ep199) has clean 72.9 and AA 0.10 (both attempts collapsed, so the AA matched by coincidence).
Since step 1 of the audit, the cluster's `results/models/convnext_base_dvd_b_linf_2_init0/` holds
attempt 2. The `dvd_b_linf_cont2to2_init0_contepoch` continuation below therefore starts from the
blessed checkpoint, and attempt 1 is in `results/models_failed/…__gradient_collapse_20260806_slurmcopy`.

## 2. The recipe: "+40 epochs, same ε, PGD 7"

**The originals are not touched.** Each one becomes the *parent* of a new run, so there is no `mv` to
`models_failed/` and no reset of an sjm or AIRCC row. The collapsed run stays in place, exactly as
trained, as the research record of what happened.

### Naming
Use the existing cont convention with target = source:
- `vit_b_cvst/linftrades_cont4to4_init1`
- `swin_b/linftrades_cont2to2_init1`
- `convnext_base_dvd_b_l2_cont2to2_init0_contepoch`
- and so on (full list in §3).

### Code change before any continuation run: skip the PGD step warmup when `continuation.enabled`

**Finding.** [train_loop.py:49](ares/utils/train_loop.py:49) sets
`att_step = attacks.attack_step * min(epoch, 5) / 5`. From scratch that is a sensible warmup. A
`continuation` run, though, restarts at epoch 0. Its training attack then works out as:
- **epoch 0: step 0.** The images are clean, apart from TRADES's tiny random start. This is a whole
  epoch of clean training on a robust model.
- **epochs 1-4:** the attack can reach only ε × epoch/5 (20%…80% of the ball). Training against an
  attack that weak is what produces gradient masking.

The convnext cont4to8 runs show it directly. The LR during epoch 0 was only 3.5e-6, yet over that
one epoch:

| | AA linf ε4 | AA linf ε8 | clean |
|---|---|---|---|
| init0: parent (`model_best_adv`, ep102) | 28.6 | 6.7 | 77.5 |
| init0: cont at ep0 | 8.6 | 0.68 | 79.8 |
| init1: parent (`last`, ep149) | 37.4 | 7.3 | 77.2 |
| init1: cont at ep1 | 7.5 | 0.29 | 78.2 |

By epoch 1, PGD-val reads an impossible 65% at ε4.
[validate.py:83](ares/utils/validate.py:83) applies the same warmup to PGD-val, so the PGD-val of
continuation epochs 1-4 is understated too. Its adversarial eval is also skipped entirely at epoch 0.

**Change (option 1, chosen 2026-10-07).**
- `ares/utils/train_loop.py`: use the full `attack_step` (or `v1_attack_step`) from epoch 0 when
  `cfg.continuation.enabled` is true. Also honour the existing `attacks.disable_attack_step_warmup`
  flag, which today only `validate.py` reads.
- `ares/utils/validate.py`: apply the same rule to the PGD-val step. For continuations, also run the
  adversarial eval at epoch 0, so each curve starts from the parent's real robustness.
- **Scope:** only runs with `continuation.enabled=true` and epoch < 5.
  - Unaffected: scratch runs, and `resume`/contepoch runs, whose epoch is ≥ 200 so the warmup is
    already over.
  - Unaffected: the three sjm continuations in flight (swin_b l2trades_cont4to8 at ep19, and
    l1trades_cont4to6 / l1trades_cont4to8, both at ep33). They are past epoch 5, so a restart after
    the change behaves identically.
  - `.agents/skills/training-runtime-optimizer/scripts/run_candidate_benchmark.py` has its own copy of
    the warmup (benchmark only). Leave it, and mention it in the skill if it matters.
- **Tests:**
  - A unit test that `train_one_epoch` passes the full step at epoch 0 when
    `continuation.enabled`, and the warmed-up step otherwise.
  - A short Botero smoke run (a few hundred steps of one recipe-A row on the ImageNet sample).
    Confirm from the log that the training attack's step equals `attack_step` at epoch 0, and that
    epoch-0 PGD-val is close to the parent's.
- **Protocol record:** every continuation trained before this change ran with the warmup. That
  covers all cont4to6/cont4to8 runs and the DVD `_resetepoch` runs. Record the change, dated, in
  `aircc_convnext_base_summary.md` and in `co_postmortem.csv` (a `step_warmup` column), so old and
  new continuation results are never compared without that note.
- **Order:** commit, push and `git pull` on the cluster **before** seeding any Task 1 or recipe-A row.
  Recipe B doesn't depend on it.

### Row recipe A: swin_b / vit_b_cvst
Copy the existing `vit_b_cvst/linftrades_cont4to6_init1` row into
`slurm_job_manager/csv/vit-b-cvst_swin-b.csv`, then change it:
- `init_mode=continuation`; `dependency_model_name=<original>`
- `continuation.enabled=true`, `continuation.use_ema=true`
- `training.epochs=40`, `lr_scheduler.lrb=0.0001`, `lr_scheduler.warmup_epochs=10`,
  `optimizer.weight_decay=0.1`, batch 256
- **`attacks.attack_it=7`** (instead of 5), with `attack_eps`, `threat_eps` = the original ε
- Epsilon schedule `warmup_ramp_fixed` with **source = target = the original ε**, keeping
  warmup 4, ramp 5-30 and fixed 31. The ε is then constant, and the code path is the same one the
  existing conts already exercise.
- `checkpointing.save_best_adv=true`

### Row recipe B: the DVD convnext models (contepoch style)
Continue the epoch counter, the same way the existing `_contepoch` rows do. A `continuation` /
`resetepoch` run restarts the epoch at 0, which **restarts the DVD age curve at the blurriest
infant stage**. Also, `model_store/zoo_excludes.csv` already keeps resetepoch DVD runs out of the
experiments.

Copy the `convnext_base_dvd_b_l2_cont4to6_init1_contepoch` row from
`aircc/aircc_job_manager/csv/`, then change it:
- `init_mode=resume`, `epoch_variant=contepoch`, `dependency_model_name=<original>`
- `resume_offset_assumed=200`, `training.epochs=240`. For the cont4to6 contepoch model: 240 → 280.
  The lifecycle shift logic sets the real target from the resumed checkpoint's epoch, +40.
- **`attacks.attack_it=7`**, `lr_scheduler.lrb=1e-4`, batch 512, `dataset.dvd.enabled=true`,
  variant `dvd-b`
- Epsilon schedule `warmup_ramp_fixed` with source = target = the original ε. Keep the existing
  shifted boundaries (warmup 204, ramp 205-230, fixed 231, each +40 for the 240→280 one).
- `continuation.use_ema=true`, `checkpointing.save_best_adv=true`

**Before seeding,** check one of each recipe with a dry `lifecycle` command build on Botero. The
build should show:
- the dependency checkpoint path
- `attack_it=7`
- the resolved epoch target: 40 epochs for recipe A, `peeked epoch + 40` for recipe B
- an unchanged DVD flag

### DB actions
- **swin/vit:** the parents are already sjm rows with status `finished` and a `best_checkpoint`.
  Only add the new CSV rows, then run `seed.py` (needs your confirmation).
- **DVD convnext:** the parents exist only in the frozen AIRCC DB, and sjm won't claim a child whose
  dependency isn't `finished` in sjm's own DB.
  1. Insert one **anchor row** per parent into `jobs.sqlite`: `status='finished'`,
     `best_checkpoint=/home/ashtomer/projects/ares/results/models/<name>/<DB-best file>`,
     `best_score`=the AIRCC score.
  2. Add the new rows to `slurm_job_manager/csv/convnext_base.csv` and seed them.

  Do the writes from the login node with python3 in one transaction (not over sshfs). Never run
  `generate_csvs.py`.
  - Check that the controller won't try to *run* an anchor row: no CSV row, `status='finished'`.
  - Check how `status.py` and the HTML generator display one.

## 3. Tasks

### Task 1: `convnext_base_linftrades_cont4to8_init{0,1}`: rerun with 7 PGD steps, without the step warmup
- **Cause:** robustness fell **within the first epoch**. Its epoch-0/1 checkpoints get AA
  0.68 / 0.29 at ε=8, against the parents' 6.7 / 7.3. Epoch 0 trained with PGD step 0 (§2, "Code
  change before any continuation run"). Training at 5 steps with a weakened step (epochs 1-4) then
  locked in the masking. `attack_it=7` alone would not have fixed epoch 0.
- **Requires:** the step-warmup code change, landed on the cluster.
- **Parents:** `convnext_base_linftrades_4_init{0,1}` are healthy (AA 28.3 / 36.1 at ε=4).
- **Steps:**
  1. Retire the two collapsed dirs:
     `mv results/models/<name> results/models_failed/<name>__co_20261007`. A rerun under the same
     name would otherwise resume the old `last.pth.tar`.
  2. Insert anchor rows for the two parents.
  3. Add both rows to the sjm CSV with `attacks.attack_it=7`, everything else unchanged, then seed
     them.
  4. Watch the first epoch's PGD-val and loss.
  5. Remove the exclude lines once both runs are `finished`.
- **Downstream:** `convnext_base_linftrades_cont8to12_init{0,1}` (AIRCC-pending, never run) stay
  parked.

### Task 2: "+40 epochs, same ε, PGD 7" for every collapsed model

| new run | parent (start = its DB-best) | recipe | epochs |
|---|---|---|---|
| swin_b/linftrades_cont4to4_init1 | swin_b/linftrades_4_init1 (model_best, ep146) | A | 40 |
| swin_b/linftrades_cont2to2_init1 | swin_b/linftrades_2_init1 (model_best, ep125) | A | 40 |
| swin_b/l2trades_cont4to4_init1 | swin_b/l2trades_4_init1 (model_best, ep82) | A | 40 |
| vit_b_cvst/linftrades_cont4to4_init1 | vit_b_cvst/linftrades_4_init1 (model_best_adv, ep142) | A | 40 |
| vit_b_cvst/l2trades_cont4to4_init1 | vit_b_cvst/l2trades_4_init1 (model_best, ep113) | A | 40 |
| convnext_base_dvd_b_linf_cont2to2_init0_contepoch | convnext_base_dvd_b_linf_2_init0 (last, ep199) | B | 200→240 |
| convnext_base_dvd_b_l2_cont2to2_init0_contepoch | convnext_base_dvd_b_l2_2_init0 (last, ep199) | B | 200→240 |
| convnext_base_dvd_b_l2_cont4to4_init1_contepoch | convnext_base_dvd_b_l2_4_init1 (last, ep199) | B | 200→240 |
| convnext_base_dvd_b_l1_cont4to4_init0_contepoch | convnext_base_dvd_b_l1_4_init0 (model_best, ep199) | B | 200→240 |
| convnext_base_dvd_b_l2_cont6to6_init1_contepoch | convnext_base_dvd_b_l2_cont4to6_init1_contepoch (model_best, ep234) | B | 240→280 |

**Evidence for and against continuing from a masked checkpoint:**
- **For:** swin_b/l2trades_cont4to6 started from the masked swin l2trades_4 (AA 0.88) and reached
  **21.0**. The vit linftrades conts reached 15.5 / 9.9 from a parent at 0.39.
- **Against:** the convnext cont4to8 runs lost robustness within one epoch of starting (Task 1).
  The cause was the PGD step warmup, which recipe A now skips (§2). Still, check epoch 0-1 PGD-val
  against the parent's AA.
- All existing conts above trained *with* the warmup, so they are not a like-for-like baseline for
  the new runs.

**If a run comes back still collapsed,** the fallback is the earlier option: resume the *original*
from its last healthy saved epoch (Appendix), or escalate with one of these:
- (a) `attack_it=10`
- (b) a random start for Madry training (a code change at `train_loop.py:79`)
- (c) a lower `lrb`

All three change the training protocol, so they are your call; none is ever applied silently.

### Downstream runs (need a decision)
These were already trained from one of the collapsed parents:

| run | state | parent |
|---|---|---|
| swin_b/linftrades_cont4to6_init1, cont4to8_init1 | finished, AA 9.47 / 9.47 | swin linftrades_4 |
| swin_b/l2trades_cont4to6_init1 | finished, AA 21.0 | swin l2trades_4 |
| **swin_b/l2trades_cont4to8_init1** | **training now** (ep 19/40 on 2026-10-07) | swin l2trades_4 |
| vit_b_cvst/linftrades_cont4to6_init1, cont4to8_init1 | finished, AA 15.5 / 9.86 | vit linftrades_4 |
| vit_b_cvst/l2trades_cont4to6_init1, cont4to8_init1 | check | vit l2trades_4 |
| convnext_base_dvd_b_l1_cont4to{6,8}_init0_contepoch | finished, AA 6.15 / 3.61 | dvd_b_l1_4_init0 |
| convnext_base_dvd_b_l2_cont4to8_init1_contepoch | finished, AA 5.76 | dvd_b_l2_4_init1 |
| convnext_base_dvd_b_l{1,2}_cont4to{6,8}_*_resetepoch | AIRCC pending, partial | as above |

Once a parent's "+40" run is healthy, you could re-derive the ε=6/8 conts from it instead of from
the collapsed original. That is a separate decision.

## 4. Keeping the results (research record)
- **Collapsed originals:** they stay in `results/models/` as the parents. Their AA CSVs, `summary.csv`
  and logs are kept. The weekly route-1 rsync already mirrors them to the QNAP.
- **`co_postmortem.csv`** (repo root, append-only). One row per model. Columns:
  - the original: name, attempt #, config (attack_it, random_start, lrb, trades_beta), collapse
    epoch from the PGD-val gap, AA and clean on its DB-best
  - the "+40" run: name and its AA and clean
  - the outcome
- **PGD curve plots:** one per original and one per "+40" run. They show clean, PGD-val and the gap
  against epoch, with train ε on a second axis. Generate them from `summary.csv` and save them as
  `co_postmortem/<flat_name>.png`.
- **`model_store`:** add `zoo_excludes.csv` patterns for the collapsed originals, so
  `models_for_experiments` stops publishing them. Once each "+40" run is healthy, decide whether it
  becomes the published model for that cell (a `model_store` override).
- **Status HTML:** the generator puts one run per norm×ε cell, and a `contEtoE` row shares its cell
  with the original. Extend `generate_status_html.py` so both show up, e.g. "original 0.00 → +40:
  X". As it stands, one of the two runs would be hidden, or would fail the coverage audit.
- When the runs finish, update this file and `aircc_convnext_base_summary.md` ("Changes from the
  original plan").

## 5. Suggested order
1. Read the copy-audit report.
2. Add the `model_store` exclusion for the collapsed originals.
3. Make the step-warmup code change (§2), with its unit test and Botero smoke run. Commit, push and
   pull it on the cluster.
4. Dry-build one recipe-A and one recipe-B command on Botero.
5. Insert the anchor rows for the AIRCC parents (DVD ×5, linftrades_4 ×2).
6. Seed the Task 2 rows and the Task 1 rows (retire the cont4to8 dirs first). Recipe-A runs are
   40-epoch jobs; recipe-B runs are 40 epochs of convnext at batch 512.
7. The AA sweep covers each new run automatically once `finished`. Fill in `co_postmortem.csv`.
8. Decide on the downstream runs.

## Appendix: fallback resume points (from the PGD-val gap; confirm with an AA probe first)
Use these only if a "+40" run fails. "periodic N" is `periodic/epoch_00NN.pth.tar`, the full saver
payload after 0-based epoch N−1. To use one, retire the original's dir, copy the checkpoint in as
`last.pth.tar`, and reset its row. sjm's `scratch` mode auto-resumes from `last.pth.tar`.

| model | collapse (PGD-val gap) | candidate |
|---|---|---|
| swin_b/linftrades_4_init1 | ep 25→30 | periodic 15 |
| swin_b/linftrades_2_init1 | ep 50→55 | periodic 30 (45 if the probe is healthy) |
| swin_b/l2trades_4_init1 | not visible in PGD-val | AA probe periodic 15…75 |
| vit_b_cvst/linftrades_4_init1 | ep 100→104 | periodic 90 |
| convnext_base_dvd_b_l2_2_init0 | ep 90→100 | QNAP periodic 90 |
| convnext_base_dvd_b_l2_4_init1 | ep 58→76, then diverges | QNAP checkpoint-58 |
| convnext_base_dvd_b_linf_2_init0 | ep 70→100 | none: scratch |
| convnext_base_dvd_b_l1_4_init0 | ep 100→110 | none: scratch |
