# Completion runs: init2/init3 retrains and continuations

*Written 2026-10-12. Replaces `co_rerun_plan.md` (the "+40 epochs at the same ε" plan, never
executed). Queue: `slurm_job_manager`, rtx_pro_6000 only.*

## Why
Twelve finished models collapsed (catastrophic overfitting or gradient masking: AA ≈ 0–12 at their
training ε, often with clean accuracy *above* the healthy reference). They are listed in
`aa_sweep/sweep_excludes.csv`, which keeps them out of the AA sweep. Instead of patching them, each
family gets a **completion run**:
- a new **init2** base run with the same hyperparameters as its init0/init1 (only the seed differs);
- continuations from it;
- for two convnext families, cross-parent continuations counted as init2/init3.

Two collapsed models are **not** retrained: `convnext_base_dvd_b_linf_2_init0` and
`convnext_base_dvd_b_l2_cont4to6_init1_contepoch`.

## The PGD step warmup in continuations
`ares/utils/train_loop.py` ramps the training PGD step as `attack_step * min(epoch, 5) / 5`. A
`continuation` run restarts its epoch counter at 0, so epoch 0 trained an already-robust model at
step 0 (clean images) and epochs 1–4 against a weakened attack. The convnext linftrades cont4to8 runs
lost robustness within that first epoch (AA at ε4: 28.6 → 8.6 and 37.4 → 7.5).

**Fix (2026-10-12):** `attacks.disable_attack_step_warmup` (default `False`, so every earlier run
is unchanged) is now honoured by training as well as validation, and is a job-manager CSV column.
With it set, training uses the full step from epoch 0 and validation also runs PGD at epoch 0.
It is `true` on the 9 continuation-mode rows below. The DVD contepoch rows resume at the parent's
epoch (≥ 5), where the ramp is already over, so they don't need it.

**Protocol note:** every continuation trained before this date ran *with* the warmup (all
cont4to6/cont4to8 runs and the DVD `_resetepoch` runs). Don't compare old and new continuations
without that note.

Botero smoke test (row 301, ImageNet sample, from the local `linftrades_4_init0/model_best_adv`):
- **flag on:** epoch-0 training step 0.00224 = (4/255)/7; epoch-0 PGD-val 36.1%;
- **flag off:** epoch-0 step 0.0, PGD-val skipped, epoch-1 step at 20% of full.

## Rules for every row
- **Seed:** `seed: null` draws a fresh random seed every launch. A copied row therefore differs
  from its template only by seed.
- **Batch size:** the CSV value is the rtx_pro_6000 batch. `manager_rtx6000.sbatch` halves it
  (`SJM_BATCH_DIVISOR=2`); these rows run only on rtx_pro_6000, so the values below are what trains.
- **Start checkpoint:** a continuation starts from its dependency's DB `best_checkpoint`.
- **Output:** `/home/ashtomer/projects/ares/results/models/<model_name>/` on the cluster, then the
  weekly rsync routes (see `model_store/README.md`). The nightly `aa_sweep` picks each run up once
  it is `finished`.

## Rows (priority ascending = claim order; `is_test` blank)

| prio | model_name | mode | starts from | epochs | bsz | PGD it | lrb | no-warmup flag |
|---|---|---|---|---|---|---|---|---|
| 301 | convnext_base_linftrades_cont4to8_init2 | continuation | `convnext_base_linftrades_4_init0` → `model_best_adv` (AA 28.6) | 40 | 256 | 7 | 1e-4 | true |
| 302 | convnext_base_linftrades_cont4to8_init3 | continuation | `convnext_base_linftrades_4_init1` → `last` (37.4) | 40 | 256 | 7 | 1e-4 | true |
| 303 | swin_b/l2trades_4_init2 | scratch | — | 150 | 256 | 3 | 5e-4 | |
| 304 | swin_b/linftrades_4_init2 | scratch | — | 150 | 256 | 3 | 5e-4 | |
| 305 | vit_b_cvst/l2trades_4_init2 | scratch | — | 150 | 256 | 3 | 5e-4 | |
| 306 | vit_b_cvst/linftrades_4_init2 | scratch | — | 150 | 256 | 3 | 5e-4 | |
| 307 | convnext_base_dvd_b_l1_4_init2 | scratch | — | 200 | 512 | 5 | 1e-4 | |
| 308 | convnext_base_dvd_b_l2_4_init2 | scratch | — | 200 | 512 | 5 | 1e-4 | |
| 311 | swin_b/linftrades_2_init2 | scratch | — | 150 | 256 | 3 | 5e-4 | |
| 312 | convnext_base_dvd_b_l2_2_init2 | scratch | — | 200 | 512 | 3 | 1e-4 | |
| 321 | convnext_base_l2_cont4to8_init2 | continuation | `convnext_base_l2_4_init0` → `model_best` (37.2) | 40 | 512 | 7 | 1e-4 | true |
| 322 | swin_b/l2trades_cont4to6_init2 | continuation | `swin_b/l2trades_4_init2` DB-best | 40 | 256 | 7 | 1e-4 | true |
| 323 | swin_b/l2trades_cont4to8_init2 | continuation | same | 40 | 256 | 7 | 1e-4 | true |
| 324 | swin_b/linftrades_cont4to6_init2 | continuation | `swin_b/linftrades_4_init2` DB-best | 40 | 256 | 7 | 1e-4 | true |
| 325 | swin_b/linftrades_cont4to8_init2 | continuation | same | 40 | 256 | 7 | 1e-4 | true |
| 326 | vit_b_cvst/l2trades_cont4to6_init2 | continuation | `vit_b_cvst/l2trades_4_init2` DB-best | 40 | 256 | 7 | 1e-4 | true |
| 327 | vit_b_cvst/l2trades_cont4to8_init2 | continuation | same | 40 | 256 | 7 | 1e-4 | true |
| 331 | convnext_base_dvd_b_l1_cont4to6_init2_contepoch | resume (contepoch) | `convnext_base_dvd_b_l1_4_init2` DB-best | best epoch + 40 | 512 | 7 | 1e-4 | |
| 332 | convnext_base_dvd_b_l1_cont4to8_init2_contepoch | resume (contepoch) | same | best epoch + 40 | 512 | 7 | 1e-4 | |
| 333 | convnext_base_dvd_b_l2_cont4to6_init2_contepoch | resume (contepoch) | `convnext_base_dvd_b_l2_4_init2` DB-best | best epoch + 40 | 512 | 7 | 1e-4 | |
| 334 | convnext_base_dvd_b_l2_cont4to8_init2_contepoch | resume (contepoch) | same | best epoch + 40 | 512 | 7 | 1e-4 | |

- **Order:** the two linftrades cont4to8 runs first, then the eps-4 bases (so their continuations
  can start as early as possible; non-DVD before DVD), then the eps-2 bases, then the other
  continuations (non-DVD before DVD).
- **Hyperparameters:** each row is a copy of its init0/init1 template, with only the names, init,
  priority, dependency and notes changed. On top of that:
  - every continuation uses PGD 7;
  - row 321 uses `lrb=1e-4`, matching init1 (init0 actually trained at 5e-4).
- **Templates:**
  - base rows keep the template's `attack_it`, 3 (default) or 5;
  - the convnext rows come from `aircc/aircc_job_manager/csv/convnext_base.csv`;
  - the swin/vit rows come from `vit-b-cvst_swin-b.csv`.
- **DVD contepoch rows:** each child's length follows its parent's DB-best epoch (the
  `resume_offset_assumed` shift). If the best is at ep 150, the child runs 150 → 190.
- **AIRCC name clash:** the AIRCC CSV already has same-named convnext init2 rows. They point at
  init2 parents and never ran. The job-manager copy is the live one.

## Anchor rows (cluster `jobs.sqlite` only, no CSV row)
Rows 301, 302 and 321 depend on AIRCC-only parents. The job manager only claims a child whose
dependency is `finished` in its own DB, so each parent gets an anchor row:
- `status='finished'`, priority 99999;
- the same shape as the existing `convnext_base_dvd_b_linf_4_init1` anchor;
- `best_checkpoint` = the frozen AIRCC DB's best, on the cluster copy.

| anchor | best_checkpoint (under `/home/ashtomer/projects/ares/results/models/`) | best_score |
|---|---|---|
| convnext_base_linftrades_4_init0 | `convnext_base_linftrades_4_init0/model_best_adv.pth.tar` | 28.613 |
| convnext_base_linftrades_4_init1 | `convnext_base_linftrades_4_init1/last.pth.tar` | 37.402 |
| convnext_base_l2_4_init0 | `convnext_base_l2_4_init0/model_best.pth.tar` | 37.207 |

## Launch
```bash
ssh slurm "cd /home/ashtomer/projects/ares && git pull"
# on the login node: insert the 3 anchors (python3, one transaction), then seed
python -m slurm_job_manager.seed --db slurm_job_manager/jobs.sqlite --arch convnext_base --init 2 --init 3
python -m slurm_job_manager.seed --db slurm_job_manager/jobs.sqlite --arch vit-b-cvst_swin-b --init 2
sbatch slurm_job_manager/slurm/manager_rtx_pro_6000.sbatch   # --array=1-200%8, rtx_pro_6000 only
```
These tasks share the account's 8 rtx_pro_6000 GPUs with the AA sweep's `aaq-pro` array. No hold
was placed on it, so training starts as those tasks free GPUs.

## Addendum (2026-10-22): cont 4→4 for the collapsed swin/vit TRADES eps-4 parents
The four collapsed init1 eps-4 TRADES parents (bsz 256, PGD 3) get a same-ε continuation, last in
the queue. Their cont4to6/4to8 children recovered from the same collapsed DB-best checkpoint.

| prio | model_name | starts from (DB best) |
|---|---|---|
| 401 | swin_b/linftrades_cont4to4_init1 | `swin_b/linftrades_4_init1/model_best` |
| 402 | swin_b/l2trades_cont4to4_init1 | `swin_b/l2trades_4_init1/model_best` |
| 403 | vit_b_cvst/linftrades_cont4to4_init1 | `vit_b_cvst/linftrades_4_init1/model_best_adv` |
| 404 | vit_b_cvst/l2trades_cont4to4_init1 | `vit_b_cvst/l2trades_4_init1/model_best` |

Each row copies its `cont4to6_init1` sibling (40 ep, lrb 1e-4, wd 0.1), with these changes:
- **PGD 5**: the same step count as the init1 cont4to6 siblings.
- **No warmups**:
  - `lr_scheduler.warmup_epochs=0`
  - `epsilon_schedule.type=fixed` at ε 4
  - `attacks.disable_attack_step_warmup=true`

They were seeded with `--arch vit-b-cvst_swin-b --init 1`, and an rtx6000 manager array
(`--array=1-200%8`) was launched alongside the rtx_pro_6000 one.

**Batch caveat:** rtx6000 halves the CSV batch (`SJM_BATCH_DIVISOR=2`), and the claim gate does
not filter by partition. So anything that array claims trains at bsz 128 (DVD rows at 256): the
4→4 rows and any pending completion row above. Check `runtime_config.yaml` → `batch_size` before
comparing a run against its siblings.

## Status page
`update-training-status-html` shows convnext_base init0/init1, plus init2/init3 rows that are
present in the job-manager CSVs. The AIRCC CSV's never-run init2 rows stay hidden.

## Botero caveat
Botero's `ares` env pairs timm 0.9.7 with torch 2.9, so `torch.load` defaults to `weights_only=True`
and rejects the OmegaConf objects in these checkpoints. Running locally needs
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. The cluster envs have timm 1.0.22, which loads with
`weights_only=False`.
