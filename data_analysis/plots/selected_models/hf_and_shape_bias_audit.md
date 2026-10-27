# Selected models: Hugging Face coverage and shape-bias checkpoint audit

Generated 2026-10-27T13:47 on Botero. Read-only audit of the 174 rows of `checkpoint_manifest.csv` (103 `in_json=True`, 71 `in_json=False`). Working files (HF metadata dump, join tables) are in the auditing session's scratchpad, not in the repo.

## TL;DR

- **HF:** 167/174 manifest rows are on HF with the identical checkpoint (same run dir and kind, and source sha256 equal). 5 rows are not on HF: all are `in_json=False` runs that `model_store/hf_excludes.csv` holds back as collapsed or overfit. 2 rows are on HF under the same name but as a **different checkpoint** (see A.1). HF also has 26 models that are not in the manifest (25 vit/swin TRADES models, plus the init2 rerun that HF publishes as `convnext_base_linftrades_cont4to8_init0`).
- **Shape bias:** all 174 manifest rows pass check (c): each filename matches its kind and each file exists. All 174 pass check (b) against the source the manifest was built from (the JSON or `models_aa_clean_by_init.csv`), and every job-DB `best_checkpoint` that names a kind agrees. The checkpoint the code loaded is therefore the one each row names. At 2026-10-27T13:47:00 the CSV covered 147/174 rows; the rest are still being scored (see B.a). **2 rows are the wrong 'winner' by the newer sources**, even though they match their own source: `convnext_base_gradnorm_l1_2_init0` and the `in_json=False` `convnext_base_linftrades_cont4to8_init0` (see B.b).

## How HF publishing works here

- The upload runs from **this repo (ares)**, not from epsilon_bounded_contstim. `model_store/hf_export.py` exports `/mnt/data4t/models_for_experiments/manifest.csv` (204 blessed models), minus the `model_store/hf_excludes.csv` patterns, as EMA `model.safetensors` + `metadata.json` into the local mirror `/mnt/data4t/hf_robustness_models`. It then runs `upload_large_folder` to **`BrainsAndMachinesLab/robustness_models`** (a private repo). `hf_proposal.py` mails proposals and gates `approve`. `verify_hf.py` checks the Hub against the zoo.
- epsilon_bounded_contstim's `models_for_adversarial_experiments.json` (the source of this manifest) is a **separate list**. It is not what gets uploaded, so the two can drift (see A.1).
- The token is valid (`whoami` = tomerash98). The listing was taken live: `list_repo_tree` plus a download of all 194 `metadata.json` files, at repo head `10f1c75c` (last modified 2026-10-22). HF holds 194 models: convnext_base 137, vit_b_cvst 29, swin_b 28.
- **How rows were matched:** HF `metadata.json` `source.model_dir`, `source.path` and `source.sha256` were compared with the manifest row's run dir, its checkpoint path, and the sha256 of that file. The sha256 comes from the model_store hash cache, which is keyed on inode+size+mtime, so it reflects the current file; all 174 rows had a cache entry. A sha256 match means the epoch matches as well.

## Part A: Hugging Face coverage

| | in_json=True | in_json=False | total |
|---|---|---|---|
| on HF, same checkpoint | 102 | 65 | 167 |
| on HF under this name, different checkpoint | 1 | 1 | 2 |
| not on HF | 0 | 5 | 5 |
| HF models not in the manifest | | | 26 |

### A.1 Manifest rows on HF

**Flagged: the name is on HF but HF holds a different checkpoint (2):**

- `convnext_base_gradnorm_l1_2_init0` (in_json=True). The manifest and shape bias use `last` (`.../convnext_base_gradnorm_l1_2_init0/last.pth.tar`). HF holds **`advbest`** (`model_best_adv.pth.tar`, epoch 126), set by a `model_store/blessing_overrides.csv` row: *"AIRCC DB recorded last.pth.tar but with advbest's score; AA linf@2 sweep CSVs: advbest 54.39 > last 54.00 > best 53.61"*. The zoo JSON still says `last`, yet reports AA 54.39, which is advbest's number.
- `convnext_base_linftrades_cont4to8_init0` (in_json=False). The manifest uses the **original collapsed run** `convnext_base_linftrades_cont4to8_init0/last.pth.tar` (AA linf@8 1.86). HF publishes the name `convnext_base_linftrades_cont4to8_init0` from the **rerun `convnext_base_linftrades_cont4to8_init2`** (`last`, AA 10.64; blessing override pinned 2026-10-22). The original run is not on HF.

**Expected aliases that match exactly** (HF name differs from the run dir, but the checkpoint is identical): `convnext_base_linftrades_cont4to8_init1` = run `..._init3` (as intended), plus the pgd5/lr1e4 reruns `convnext_base_{l1,l2,linf}_cont4to6_init0` and `convnext_base_{l1,linf}_cont4to8_init0`, which are published under the plain names.

**On HF with the same checkpoint (167),** by family. `[kind]` is the checkpoint kind:

- **baselines** (10): `convnext_base_baseline_init0` [best], `convnext_base_baseline_init1` [best], `convnext_base_dvd_b_baseline_init0` [best], `convnext_base_dvd_b_baseline_init1` [best], `convnext_base_v1_clean_init0` [last], `convnext_base_v1_clean_init1` [best], `convnext_base_v1_noise_init0` [best], `convnext_base_v1_noise_init1` [best], `vit_b_cvst_baseline_init1` [last], `swin_b_baseline_init1` [best]
- **convnext_base madry_l1** (10): `convnext_base_l1_1_init0` [best], `convnext_base_l1_1_init1` [best], `convnext_base_l1_2_init0` [best], `convnext_base_l1_2_init1` [last], `convnext_base_l1_4_init0` [advbest], `convnext_base_l1_4_init1` [last], `convnext_base_l1_cont4to6_init0` (run `convnext_base_l1_cont4to6_lr1e4_pgd5_init0`) [last], `convnext_base_l1_cont4to6_init1` [last], `convnext_base_l1_cont4to8_init0` (run `convnext_base_l1_cont4to8_pgd5_init0`) [last], `convnext_base_l1_cont4to8_init1` [last]
- **convnext_base madry_l2** (10): `convnext_base_l2_1_init0` [advbest], `convnext_base_l2_1_init1` [best], `convnext_base_l2_2_init0` [advbest], `convnext_base_l2_2_init1` [best], `convnext_base_l2_4_init0` [best], `convnext_base_l2_4_init1` [last], `convnext_base_l2_cont4to6_init0` (run `convnext_base_l2_cont4to6_lr1e4_pgd5_init0`) [last], `convnext_base_l2_cont4to6_init1` [last], `convnext_base_l2_cont4to8_init0` [last], `convnext_base_l2_cont4to8_init1` [last]
- **convnext_base madry_linf** (10): `convnext_base_linf_1_init0` [best], `convnext_base_linf_1_init1` [best], `convnext_base_linf_2_init0` [advbest], `convnext_base_linf_2_init1` [last], `convnext_base_linf_4_init0` [last], `convnext_base_linf_4_init1` [last], `convnext_base_linf_cont4to6_init0` (run `convnext_base_linf_cont4to6_pgd5_init0`) [advbest], `convnext_base_linf_cont4to6_init1` [last], `convnext_base_linf_cont4to8_init0` (run `convnext_base_linf_cont4to8_pgd5_init0`) [last], `convnext_base_linf_cont4to8_init1` [last]
- **convnext_base trades_l1** (10): `convnext_base_l1trades_1_init0` [last], `convnext_base_l1trades_1_init1` [advbest], `convnext_base_l1trades_2_init0` [last], `convnext_base_l1trades_2_init1` [last], `convnext_base_l1trades_4_init0` [best], `convnext_base_l1trades_4_init1` [last], `convnext_base_l1trades_cont4to6_init0` [last], `convnext_base_l1trades_cont4to6_init1` [last], `convnext_base_l1trades_cont4to8_init0` [last], `convnext_base_l1trades_cont4to8_init1` [last]
- **convnext_base trades_l2** (10): `convnext_base_l2trades_1_init0` [advbest], `convnext_base_l2trades_1_init1` [last], `convnext_base_l2trades_2_init0` [advbest], `convnext_base_l2trades_2_init1` [last], `convnext_base_l2trades_4_init0` [last], `convnext_base_l2trades_4_init1` [advbest], `convnext_base_l2trades_cont4to6_init0` [last], `convnext_base_l2trades_cont4to6_init1` [last], `convnext_base_l2trades_cont4to8_init0` [last], `convnext_base_l2trades_cont4to8_init1` [best]
- **convnext_base trades_linf** (9): `convnext_base_linftrades_1_init0` [last], `convnext_base_linftrades_1_init1` [last], `convnext_base_linftrades_2_init0` [last], `convnext_base_linftrades_2_init1` [last], `convnext_base_linftrades_4_init0` [advbest], `convnext_base_linftrades_4_init1` [last], `convnext_base_linftrades_cont4to6_init0` [last], `convnext_base_linftrades_cont4to6_init1` [last], `convnext_base_linftrades_cont4to8_init1` (run `convnext_base_linftrades_cont4to8_init3`) [last]
- **convnext_base dvdb_madry_l1** (9): `convnext_base_dvd_b_l1_1_init0` [best], `convnext_base_dvd_b_l1_1_init1` [last], `convnext_base_dvd_b_l1_2_init0` [last], `convnext_base_dvd_b_l1_2_init1` [best], `convnext_base_dvd_b_l1_4_init1` [advbest], `convnext_base_dvd_b_l1_cont4to6_init0_contepoch` [last], `convnext_base_dvd_b_l1_cont4to6_init1_contepoch` [best], `convnext_base_dvd_b_l1_cont4to8_init0_contepoch` [last], `convnext_base_dvd_b_l1_cont4to8_init1_contepoch` [last]
- **convnext_base dvdb_madry_l2** (7): `convnext_base_dvd_b_l2_1_init0` [best], `convnext_base_dvd_b_l2_1_init1` [advbest], `convnext_base_dvd_b_l2_2_init1` [best], `convnext_base_dvd_b_l2_4_init0` [last], `convnext_base_dvd_b_l2_cont4to6_init0_contepoch` [best], `convnext_base_dvd_b_l2_cont4to8_init0_contepoch` [best], `convnext_base_dvd_b_l2_cont4to8_init1_contepoch` [best]
- **convnext_base dvdb_madry_linf** (9): `convnext_base_dvd_b_linf_1_init0` [last], `convnext_base_dvd_b_linf_1_init1` [last], `convnext_base_dvd_b_linf_2_init1` [best], `convnext_base_dvd_b_linf_4_init0` [last], `convnext_base_dvd_b_linf_4_init1` [last], `convnext_base_dvd_b_linf_cont4to6_init0_contepoch` [last], `convnext_base_dvd_b_linf_cont4to6_init1_contepoch` [last], `convnext_base_dvd_b_linf_cont4to8_init0_contepoch` [last], `convnext_base_dvd_b_linf_cont4to8_init1_contepoch` [last]
- **convnext_base dvdb_trades_l1** (10): `convnext_base_dvd_b_l1trades_1_init0` [last], `convnext_base_dvd_b_l1trades_1_init1` [advbest], `convnext_base_dvd_b_l1trades_2_init0` [best], `convnext_base_dvd_b_l1trades_2_init1` [last], `convnext_base_dvd_b_l1trades_4_init0` [advbest], `convnext_base_dvd_b_l1trades_4_init1` [best], `convnext_base_dvd_b_l1trades_cont4to6_init0_contepoch` [last], `convnext_base_dvd_b_l1trades_cont4to6_init1_contepoch` [advbest], `convnext_base_dvd_b_l1trades_cont4to8_init0_contepoch` [last], `convnext_base_dvd_b_l1trades_cont4to8_init1_contepoch` [advbest]
- **convnext_base dvdb_trades_l2** (10): `convnext_base_dvd_b_l2trades_1_init0` [best], `convnext_base_dvd_b_l2trades_1_init1` [best], `convnext_base_dvd_b_l2trades_2_init0` [advbest], `convnext_base_dvd_b_l2trades_2_init1` [advbest], `convnext_base_dvd_b_l2trades_4_init0` [advbest], `convnext_base_dvd_b_l2trades_4_init1` [advbest], `convnext_base_dvd_b_l2trades_cont4to6_init0_contepoch` [advbest], `convnext_base_dvd_b_l2trades_cont4to6_init1_contepoch` [advbest], `convnext_base_dvd_b_l2trades_cont4to8_init0_contepoch` [advbest], `convnext_base_dvd_b_l2trades_cont4to8_init1_contepoch` [advbest]
- **convnext_base dvdb_trades_linf** (10): `convnext_base_dvd_b_linftrades_1_init0` [best], `convnext_base_dvd_b_linftrades_1_init1` [advbest], `convnext_base_dvd_b_linftrades_2_init0` [best], `convnext_base_dvd_b_linftrades_2_init1` [best], `convnext_base_dvd_b_linftrades_4_init0` [best], `convnext_base_dvd_b_linftrades_4_init1` [best], `convnext_base_dvd_b_linftrades_cont4to6_init0_contepoch` [last], `convnext_base_dvd_b_linftrades_cont4to6_init1_contepoch` [last], `convnext_base_dvd_b_linftrades_cont4to8_init0_contepoch` [last], `convnext_base_dvd_b_linftrades_cont4to8_init1_contepoch` [last]
- **convnext_base v1_madry_l2** (10): `convnext_base_v1_l2_1_init0` [advbest], `convnext_base_v1_l2_1_init1` [best], `convnext_base_v1_l2_2_init0` [best], `convnext_base_v1_l2_2_init1` [last], `convnext_base_v1_l2_4_init0` [last], `convnext_base_v1_l2_4_init1` [last], `convnext_base_v1_l2_cont4to6_init0` [last], `convnext_base_v1_l2_cont4to6_init1` [advbest], `convnext_base_v1_l2_cont4to8_init0` [last], `convnext_base_v1_l2_cont4to8_init1` [last]
- **convnext_base gradnorm_l1** (3): `convnext_base_gradnorm_l1_1_init0` [best], `convnext_base_gradnorm_l1_1_init1` [last], `convnext_base_gradnorm_l1_2_init1` [advbest]
- **vit_b_cvst madry_l1** (5): `vit_b_cvst_l1_1_init1` [advbest], `vit_b_cvst_l1_2_init1` [last], `vit_b_cvst_l1_4_init1` [last], `vit_b_cvst_l1_cont4to6_init1` [last], `vit_b_cvst_l1_cont4to8_init1` [last]
- **vit_b_cvst madry_l2** (5): `vit_b_cvst_l2_1_init1` [advbest], `vit_b_cvst_l2_2_init1` [advbest], `vit_b_cvst_l2_4_init1` [advbest], `vit_b_cvst_l2_cont4to6_init1` [last], `vit_b_cvst_l2_cont4to8_init1` [last]
- **vit_b_cvst madry_linf** (5): `vit_b_cvst_linf_1_init1` [best], `vit_b_cvst_linf_2_init1` [best], `vit_b_cvst_linf_4_init1` [advbest], `vit_b_cvst_linf_cont4to6_init1` [last], `vit_b_cvst_linf_cont4to8_init1` [advbest]
- **swin_b madry_l1** (5): `swin_b_l1_1_init1` [advbest], `swin_b_l1_2_init1` [last], `swin_b_l1_4_init1` [advbest], `swin_b_l1_cont4to6_init1` [last], `swin_b_l1_cont4to8_init1` [last]
- **swin_b madry_l2** (5): `swin_b_l2_1_init1` [last], `swin_b_l2_2_init1` [advbest], `swin_b_l2_4_init1` [last], `swin_b_l2_cont4to6_init1` [last], `swin_b_l2_cont4to8_init1` [best]
- **swin_b madry_linf** (5): `swin_b_linf_1_init1` [last], `swin_b_linf_2_init1` [best], `swin_b_linf_4_init1` [best], `swin_b_linf_cont4to6_init1` [last], `swin_b_linf_cont4to8_init1` [last]

### A.2 Manifest rows not on HF (5, all in_json=False)

Each one is blessed in `models_for_experiments` (same target) but excluded by `model_store/hf_excludes.csv`:

- convnext_base dvdb_madry_l1: `convnext_base_dvd_b_l1_4_init0` [best]: aa_sweep/sweep_excludes.csv: 2026-10-07 catastrophic overfitting (AA 4.98); replaced by convnext_base_dvd_b_l1_4_init2 -- see completion_runs_plan.md
- convnext_base dvdb_madry_l2: `convnext_base_dvd_b_l2_2_init0` [last]: aa_sweep/sweep_excludes.csv: 2026-10-07 catastrophic overfitting (AA 0.20); replaced by convnext_base_dvd_b_l2_2_init2 -- see completion_runs_plan.md
- convnext_base dvdb_madry_l2: `convnext_base_dvd_b_l2_4_init1` [last]: aa_sweep/sweep_excludes.csv: 2026-10-07 collapse/divergence ~ep 76-90 (AA 8.79); replaced by convnext_base_dvd_b_l2_4_init2 -- see completion_runs_plan.md
- convnext_base dvdb_madry_l2: `convnext_base_dvd_b_l2_cont4to6_init1_contepoch` [best]: aa_sweep/sweep_excludes.csv: 2026-10-07 collapsed (AA 0.10); not retrained -- see completion_runs_plan.md
- convnext_base dvdb_madry_linf: `convnext_base_dvd_b_linf_2_init0` [last]: aa_sweep/sweep_excludes.csv: 2026-10-07 catastrophic overfitting (AA 0.10); not retrained -- see completion_runs_plan.md

The two A.1-flagged rows are also absent from HF *as these exact checkpoints*, so **7 manifest checkpoints in total are not on HF**.

### A.3 HF models not in the manifest (26)

- **convnext_base trades** (1): `convnext_base_linftrades_cont4to8_init0` (run `convnext_base_linftrades_cont4to8_init2`) [last]
- **swin_b trades** (12): `swin_b_l2trades_cont4to6_init1` [last], `swin_b_l2trades_1_init1` [advbest], `swin_b_l2trades_2_init1` [last], `swin_b_l2trades_cont4to8_init1` [last], `swin_b_linftrades_cont4to6_init1` [last], `swin_b_linftrades_cont4to8_init1` [last], `swin_b_linftrades_1_init1` [advbest], `swin_b_l1trades_2_init1` [best], `swin_b_l1trades_cont4to6_init1` [last], `swin_b_l1trades_cont4to8_init1` [last], `swin_b_l1trades_4_init1` [last], `swin_b_l1trades_1_init1` [last]
- **vit_b_cvst trades** (13): `vit_b_cvst_l2trades_cont4to8_init1` [last], `vit_b_cvst_l2trades_cont4to6_init1` [last], `vit_b_cvst_l2trades_2_init1` [best], `vit_b_cvst_l2trades_1_init1` [best], `vit_b_cvst_linftrades_2_init1` [best], `vit_b_cvst_linftrades_1_init1` [advbest], `vit_b_cvst_linftrades_cont4to8_init1` [last], `vit_b_cvst_linftrades_cont4to6_init1` [last], `vit_b_cvst_l1trades_2_init1` [last], `vit_b_cvst_l1trades_cont4to6_init1` [last], `vit_b_cvst_l1trades_1_init1` [advbest], `vit_b_cvst_l1trades_4_init1` [last], `vit_b_cvst_l1trades_cont4to8_init1` [last]

The vit_b_cvst/swin_b TRADES runs are on HF, but `selected_models.py` defines only madry families for those archs. The convnext extra is the init2 rerun noted in A.1.

## Part B: shape-bias checkpoint correctness

Checks per manifest row:
- **(a)** `shape_bias_results_botero.csv` has exactly one row for (run_dir, kind).
- **(b)** The kind equals its source: the zoo JSON `checkpoint_kind` (matched on the exact checkpoint path) for in_json=True, or `models_aa_clean_by_init.csv` `initN_ckpt` for in_json=False. It is then cross-checked against `jobs.best_checkpoint` in the frozen AIRCC DB and the live sjm DB (both opened `mode=ro&immutable=1`; looked up by run dir, then `arch/` form, then published name).
- **(c)** The manifest path's filename is that kind, and the file exists.

`shape_bias_selected.py` loads `Selected.checkpoint` (that exact path) through `load_model`, which uses `state_dict_ema` when present, the same EMA weights HF publishes. It writes `model_name=run_dir`, `checkpoint=kind`. Passing (b)+(c) therefore means the CSV row was scored on the checkpoint the row names.

| check | pass |
|---|---|
| (a) CSV row present (as of 2026-10-27T13:47:00) | 147/174 |
| (b) kind = source (JSON / by_init CSV) | 174/174 |
| (b) agrees with every DB best_checkpoint that names a kind | 174/174 |
| (c) filename matches kind | 174/174 |
| (c) file exists | 174/174 |

No CSV row falls outside the manifest. Duplicate (run_dir, kind) rows in the CSV: 0.

### B.a Rows without a CSV score yet

27 rows. Two `shape_bias_selected` processes from the other session were still appending at audit time (newest CSV row 2026-10-27T13:47:00). These rows are **pending, not failed**; re-check after those jobs exit.

- **convnext_base dvdb_madry_linf**: `convnext_base_dvd_b_linf_1_init1` [last], `convnext_base_dvd_b_linf_2_init0` [last], `convnext_base_dvd_b_linf_4_init1` [last], `convnext_base_dvd_b_linf_cont4to6_init1_contepoch` [last], `convnext_base_dvd_b_linf_cont4to8_init1_contepoch` [last]
- **convnext_base dvdb_trades_l1**: `convnext_base_dvd_b_l1trades_1_init1` [advbest], `convnext_base_dvd_b_l1trades_2_init1` [last], `convnext_base_dvd_b_l1trades_4_init1` [best], `convnext_base_dvd_b_l1trades_cont4to6_init1_contepoch` [advbest], `convnext_base_dvd_b_l1trades_cont4to8_init1_contepoch` [advbest]
- **convnext_base dvdb_trades_l2**: `convnext_base_dvd_b_l2trades_1_init1` [best], `convnext_base_dvd_b_l2trades_2_init1` [advbest], `convnext_base_dvd_b_l2trades_4_init1` [advbest], `convnext_base_dvd_b_l2trades_cont4to6_init1_contepoch` [advbest], `convnext_base_dvd_b_l2trades_cont4to8_init1_contepoch` [advbest]
- **convnext_base dvdb_trades_linf**: `convnext_base_dvd_b_linftrades_1_init1` [advbest], `convnext_base_dvd_b_linftrades_2_init0` [best], `convnext_base_dvd_b_linftrades_4_init1` [best], `convnext_base_dvd_b_linftrades_cont4to6_init1_contepoch` [last], `convnext_base_dvd_b_linftrades_cont4to8_init1_contepoch` [last]
- **convnext_base v1_madry_l2**: `convnext_base_v1_l2_1_init1` [best], `convnext_base_v1_l2_2_init1` [last], `convnext_base_v1_l2_4_init1` [last], `convnext_base_v1_l2_cont4to6_init1` [advbest], `convnext_base_v1_l2_cont4to8_init1` [last]
- **convnext_base gradnorm_l1**: `convnext_base_gradnorm_l1_1_init1` [last], `convnext_base_gradnorm_l1_2_init1` [advbest]

Note: the two scoring processes started at different times, and each computed its to-do list once at startup. If the first (full) run's list also contains in_json=False rows, it will score them a second time and append duplicates. The scores should be identical (see B.d), but de-dupe before plotting.

### B.b Kind 'winner' disagreements between sources

Each row matches the source it was built from. The disagreements come from the newer `blessing_overrides.csv` / `models_for_experiments` / HF picks:

1. **`convnext_base_gradnorm_l1_2_init0` (in_json=True, scored `last`).** The JSON, `by_init` (`init0_ckpt=last`) and AIRCC DB (`best_checkpoint=last.pth.tar`) all say `last`. `blessing_overrides.csv`, `models_for_experiments/manifest.csv` and HF say `advbest`, because the AIRCC DB stored last.pth.tar alongside advbest's score (54.39). The JSON's AA 54.39 belongs to `advbest`, so the report pairs an advbest AA number with a `last` shape-bias score. **Decide which kind is the model, then re-score that kind if it is advbest.**
2. **`convnext_base_linftrades_cont4to8_init0` (in_json=False, scored `last` of the original run).** `models_aa_clean_by_init.csv` is dated 2026-10-12 and still lists the collapsed original (AA 1.86) as init0, with init2/init3 `pending`. The zoo JSON (2026-10-22) itself says *"The init0 rerun (init2) scores 10.64"*, and `models_for_experiments` + HF use `convnext_base_linftrades_cont4to8_init2/last.pth.tar`. The init1 side is already the init3 rerun, so this one family compares a rerun (init3) with a collapsed original. Shape bias 0.7328 is for the collapsed model. **Swap in init2 if the comparison should use the rerun.**
3. **The 5 HF-excluded in_json=False runs (A.2)** are consistent across every source: they are the blessed checkpoints of those run dirs. They are, however, known collapsed or overfit runs, so their shape-bias scores describe failed models. `convnext_base_dvd_b_l2_2_init2`, the replacement for `dvd_b_l2_2_init0`, exists in `/mnt/data4t/models`. The others were "not retrained" or have no replacement on Botero.

DB coverage: 116 rows have an AIRCC `best_checkpoint` and 65 have an sjm one (33 have both, and every row has at least one). Where both exist they agree. AIRCC rows marked `running`/`pending` with no best_checkpoint are runs the frozen archive caught mid-flight; sjm finished them. `convnext_base_linftrades_cont4to8_init3` has no AIRCC row of its own: the AIRCC `..._init1` row points at the original init1 dir, and sjm's row for init3 says `last`, which matches.

### B.c Checkpoint files

All 174 manifest paths exist under `/mnt/data4t/models/<arch>/<run_dir>/`, and every filename maps to its kind (`model_best.pth.tar`=best, `last.pth.tar`=last, `model_best_adv.pth.tar`=advbest). No row fell back to an alternate filename.

### B.d Re-score check

These rows were re-scored with the same functions `shape_bias_selected.py` uses (`resolve` → `load_model` → `_score_checkpoint` → `summarize_image_rows`), in a print-only copy of the script that writes nothing to the CSV:

| run_dir | kind | CSV score / score_max | re-score | identical |
|---|---|---|---|---|
| `convnext_base_baseline_init0` | best | 0.3802024746906636 / 0.3760479041916167 | 0.38020247469066365 / 0.37604790419161677 | yes |
| `swin_b_l1_4_init1` | advbest | 0.7060333761232349 / 0.7082191780821918 | 0.7060333761232349 / 0.7082191780821918 | yes |
| `convnext_base_linftrades_cont4to8_init3` | last | 0.7183908045977011 / 0.7280701754385965 | 0.7183908045977011 / 0.7280701754385965 | yes |
| `vit_b_cvst_l1_1_init1` | advbest | 0.6127819548872181 / 0.6187245590230664 | 0.6127819548872181 / 0.6187245590230664 | yes |
| `convnext_base_dvd_b_l2trades_2_init0` | advbest | 0.7481840193704601 / 0.7575757575757576 | 0.7481840193704601 / 0.7575757575757576 | yes |

## Appendix: per-row Part B results

`src` = kind from JSON (in_json) or by_init CSV. `aircc`/`sjm` = kind from that DB's best_checkpoint. OK = (b)+(c) pass; FLAG = pass but disputed (B.b); PENDING = no CSV row yet.

| family | run_dir | kind | in_json | src | aircc | sjm | score | result |
|---|---|---|---|---|---|---|---|---|
| baselines | `convnext_base_baseline_init0` | best | True | best | best | best | 0.3802 | OK |
| baselines | `convnext_base_baseline_init1` | best | False | best | best | best | 0.3746 | OK |
| baselines | `convnext_base_dvd_b_baseline_init0` | best | True | best | best |  | 0.5445 | OK |
| baselines | `convnext_base_dvd_b_baseline_init1` | best | False | best | best |  | 0.5691 | OK |
| baselines | `convnext_base_v1_clean_init0` | last | True | last | running:no best_ckpt | last | 0.3801 | OK |
| baselines | `convnext_base_v1_clean_init1` | best | False | best | running:no best_ckpt | best | 0.3952 | OK |
| baselines | `convnext_base_v1_noise_init0` | best | True | best | best |  | 0.5080 | OK |
| baselines | `convnext_base_v1_noise_init1` | best | False | best | running:no best_ckpt | best | 0.5573 | OK |
| baselines | `vit_b_cvst_baseline_init1` | last | True | last |  | last | 0.3493 | OK |
| baselines | `swin_b_baseline_init1` | best | True | best |  | best | 0.2660 | OK |
| convnext_base madry_l1 | `convnext_base_l1_1_init0` | best | True | best | best |  | 0.6100 | OK |
| convnext_base madry_l1 | `convnext_base_l1_1_init1` | best | False | best | best |  | 0.6075 | OK |
| convnext_base madry_l1 | `convnext_base_l1_2_init0` | best | True | best | best |  | 0.6543 | OK |
| convnext_base madry_l1 | `convnext_base_l1_2_init1` | last | False | last | last |  | 0.6844 | OK |
| convnext_base madry_l1 | `convnext_base_l1_4_init0` | advbest | True | advbest | advbest |  | 0.7139 | OK |
| convnext_base madry_l1 | `convnext_base_l1_4_init1` | last | False | last | last |  | 0.6913 | OK |
| convnext_base madry_l1 | `convnext_base_l1_cont4to6_lr1e4_pgd5_init0` | last | True | last | last |  | 0.7352 | OK |
| convnext_base madry_l1 | `convnext_base_l1_cont4to6_init1` | last | False | last | last |  | 0.7106 | OK |
| convnext_base madry_l1 | `convnext_base_l1_cont4to8_pgd5_init0` | last | True | last | last |  | 0.7544 | OK |
| convnext_base madry_l1 | `convnext_base_l1_cont4to8_init1` | last | False | last | last |  | 0.7275 | OK |
| convnext_base madry_l2 | `convnext_base_l2_1_init0` | advbest | True | advbest | advbest |  | 0.6971 | OK |
| convnext_base madry_l2 | `convnext_base_l2_1_init1` | best | False | best | best |  | 0.6885 | OK |
| convnext_base madry_l2 | `convnext_base_l2_2_init0` | advbest | True | advbest | advbest |  | 0.7621 | OK |
| convnext_base madry_l2 | `convnext_base_l2_2_init1` | best | False | best | best |  | 0.7497 | OK |
| convnext_base madry_l2 | `convnext_base_l2_4_init0` | best | True | best | best | best | 0.8138 | OK |
| convnext_base madry_l2 | `convnext_base_l2_4_init1` | last | False | last | last |  | 0.7805 | OK |
| convnext_base madry_l2 | `convnext_base_l2_cont4to6_lr1e4_pgd5_init0` | last | True | last | last |  | 0.8339 | OK |
| convnext_base madry_l2 | `convnext_base_l2_cont4to6_init1` | last | False | last | last |  | 0.8118 | OK |
| convnext_base madry_l2 | `convnext_base_l2_cont4to8_init0` | last | False | last | last |  | 0.7939 | OK |
| convnext_base madry_l2 | `convnext_base_l2_cont4to8_init1` | last | True | last | last |  | 0.8287 | OK |
| convnext_base madry_linf | `convnext_base_linf_1_init0` | best | True | best | best |  | 0.6718 | OK |
| convnext_base madry_linf | `convnext_base_linf_1_init1` | best | False | best | best |  | 0.6557 | OK |
| convnext_base madry_linf | `convnext_base_linf_2_init0` | advbest | True | advbest | advbest |  | 0.7153 | OK |
| convnext_base madry_linf | `convnext_base_linf_2_init1` | last | False | last | last |  | 0.7061 | OK |
| convnext_base madry_linf | `convnext_base_linf_4_init0` | last | True | last | last |  | 0.7461 | OK |
| convnext_base madry_linf | `convnext_base_linf_4_init1` | last | False | last | last |  | 0.7541 | OK |
| convnext_base madry_linf | `convnext_base_linf_cont4to6_pgd5_init0` | advbest | True | advbest | advbest |  | 0.7761 | OK |
| convnext_base madry_linf | `convnext_base_linf_cont4to6_init1` | last | False | last | last |  | 0.7808 | OK |
| convnext_base madry_linf | `convnext_base_linf_cont4to8_pgd5_init0` | last | True | last | last |  | 0.7646 | OK |
| convnext_base madry_linf | `convnext_base_linf_cont4to8_init1` | last | False | last | last |  | 0.7861 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_1_init0` | last | True | last | last |  | 0.5226 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_1_init1` | advbest | False | advbest | advbest |  | 0.5315 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_2_init0` | last | True | last | last |  | 0.5633 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_2_init1` | last | False | last | last |  | 0.6286 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_4_init0` | best | True | best | best |  | 0.6422 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_4_init1` | last | False | last | last |  | 0.6082 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_cont4to6_init0` | last | True | last | last |  | 0.6790 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_cont4to6_init1` | last | False | last | last |  | 0.6521 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_cont4to8_init0` | last | True | last | last |  | 0.7025 | OK |
| convnext_base trades_l1 | `convnext_base_l1trades_cont4to8_init1` | last | False | last | last |  | 0.6798 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_1_init0` | advbest | True | advbest | advbest |  | 0.6282 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_1_init1` | last | False | last | last |  | 0.6211 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_2_init0` | advbest | True | advbest | advbest |  | 0.6701 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_2_init1` | last | False | last | last |  | 0.6877 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_4_init0` | last | True | last | last |  | 0.7230 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_4_init1` | advbest | False | advbest | advbest |  | 0.7530 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_cont4to6_init0` | last | True | last | last |  | 0.7285 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_cont4to6_init1` | last | False | last | last |  | 0.7646 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_cont4to8_init0` | last | True | last | last |  | 0.7275 | OK |
| convnext_base trades_l2 | `convnext_base_l2trades_cont4to8_init1` | best | False | best | best |  | 0.7534 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_1_init0` | last | True | last | last |  | 0.5858 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_1_init1` | last | False | last | last |  | 0.6013 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_2_init0` | last | True | last | last |  | 0.6597 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_2_init1` | last | False | last | last |  | 0.6294 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_4_init0` | advbest | False | advbest | advbest | advbest | 0.6946 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_4_init1` | last | True | last | last | last | 0.6950 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_cont4to6_init0` | last | True | last | last |  | 0.7144 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_cont4to6_init1` | last | False | last | last |  | 0.7187 | OK |
| convnext_base trades_linf | `convnext_base_linftrades_cont4to8_init0` | last | False | last | last |  | 0.7328 | FLAG |
| convnext_base trades_linf | `convnext_base_linftrades_cont4to8_init3` | last | True | last | last | last | 0.7184 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_1_init0` | best | True | best | best |  | 0.7018 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_1_init1` | last | False | last | last |  | 0.7243 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_2_init0` | last | True | last | last |  | 0.7802 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_2_init1` | best | False | best | best |  | 0.7507 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_4_init0` | best | False | best | best |  | 0.6758 | FLAG |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_4_init1` | advbest | True | advbest | advbest |  | 0.7643 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_cont4to6_init0_contepoch` | last | False | last | last |  | 0.6910 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_cont4to6_init1_contepoch` | best | True | best | best |  | 0.7597 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_cont4to8_init0_contepoch` | last | False | last | last |  | 0.6992 | OK |
| convnext_base dvdb_madry_l1 | `convnext_base_dvd_b_l1_cont4to8_init1_contepoch` | last | True | last | last |  | 0.7676 | OK |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_1_init0` | best | True | best | best |  | 0.7907 | OK |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_1_init1` | advbest | False | advbest | advbest |  | 0.7716 | OK |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_2_init0` | last | False | last | last |  | 0.7782 | FLAG |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_2_init1` | best | True | best | best |  | 0.7886 | OK |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_4_init0` | last | True | last | last |  | 0.8103 | OK |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_4_init1` | last | False | last | last |  | 0.7910 | FLAG |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_cont4to6_init0_contepoch` | best | True | best | best |  | 0.8166 | OK |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_cont4to6_init1_contepoch` | best | False | best | best |  | 0.7081 | FLAG |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_cont4to8_init0_contepoch` | best | True | best | best |  | 0.8300 | OK |
| convnext_base dvdb_madry_l2 | `convnext_base_dvd_b_l2_cont4to8_init1_contepoch` | best | False | best | best |  | 0.7955 | OK |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_1_init0` | last | True | last | last |  | 0.7385 | OK |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_1_init1` | last | False | last | last |  |  | PENDING |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_2_init0` | last | False | last | last |  |  | PENDING |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_2_init1` | best | True | best | best |  | 0.7877 | OK |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_4_init0` | last | True | last | last |  | 0.7947 | OK |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_4_init1` | last | False | last | last | last |  | PENDING |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_cont4to6_init0_contepoch` | last | True | last | last |  | 0.8141 | OK |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_cont4to6_init1_contepoch` | last | False | last | running:no best_ckpt | last |  | PENDING |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_cont4to8_init0_contepoch` | last | True | last | last |  | 0.8293 | OK |
| convnext_base dvdb_madry_linf | `convnext_base_dvd_b_linf_cont4to8_init1_contepoch` | last | False | last | running:no best_ckpt | last |  | PENDING |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_1_init0` | last | True | last | last |  | 0.6762 | OK |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_1_init1` | advbest | False | advbest | advbest |  |  | PENDING |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_2_init0` | best | True | best | best |  | 0.6788 | OK |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_2_init1` | last | False | last | last |  |  | PENDING |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_4_init0` | advbest | True | advbest | advbest |  | 0.6936 | OK |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_4_init1` | best | False | best | running:no best_ckpt | best |  | PENDING |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_cont4to6_init0_contepoch` | last | True | last | last |  | 0.7235 | OK |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_cont4to6_init1_contepoch` | advbest | False | advbest | pending:no best_ckpt | advbest |  | PENDING |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_cont4to8_init0_contepoch` | last | True | last | last |  | 0.7326 | OK |
| convnext_base dvdb_trades_l1 | `convnext_base_dvd_b_l1trades_cont4to8_init1_contepoch` | advbest | False | advbest | pending:no best_ckpt | advbest |  | PENDING |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_1_init0` | best | True | best | best |  | 0.7240 | OK |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_1_init1` | best | False | best | best |  |  | PENDING |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_2_init0` | advbest | True | advbest | advbest |  | 0.7482 | OK |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_2_init1` | advbest | False | advbest | running:no best_ckpt | advbest |  | PENDING |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_4_init0` | advbest | True | advbest | advbest |  | 0.7669 | OK |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_4_init1` | advbest | False | advbest | running:no best_ckpt | advbest |  | PENDING |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_cont4to6_init0_contepoch` | advbest | True | advbest | advbest |  | 0.7717 | OK |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_cont4to6_init1_contepoch` | advbest | False | advbest | pending:no best_ckpt | advbest |  | PENDING |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_cont4to8_init0_contepoch` | advbest | True | advbest | advbest |  | 0.7670 | OK |
| convnext_base dvdb_trades_l2 | `convnext_base_dvd_b_l2trades_cont4to8_init1_contepoch` | advbest | False | advbest | pending:no best_ckpt | advbest |  | PENDING |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_1_init0` | best | True | best | best |  | 0.7014 | OK |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_1_init1` | advbest | False | advbest | running:no best_ckpt | advbest |  | PENDING |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_2_init0` | best | False | best | running:no best_ckpt | best |  | PENDING |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_2_init1` | best | True | best | best |  | 0.7413 | OK |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_4_init0` | best | True | best | running:no best_ckpt | best | 0.7819 | OK |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_4_init1` | best | False | best | running:no best_ckpt | best |  | PENDING |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_cont4to6_init0_contepoch` | last | True | last | pending:no best_ckpt | last | 0.7836 | OK |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_cont4to6_init1_contepoch` | last | False | last | pending:no best_ckpt | last |  | PENDING |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_cont4to8_init0_contepoch` | last | True | last | pending:no best_ckpt | last | 0.7961 | OK |
| convnext_base dvdb_trades_linf | `convnext_base_dvd_b_linftrades_cont4to8_init1_contepoch` | last | False | last | pending:no best_ckpt | last |  | PENDING |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_1_init0` | advbest | True | advbest | advbest |  | 0.6524 | OK |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_1_init1` | best | False | best | running:no best_ckpt | best |  | PENDING |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_2_init0` | best | True | best | best |  | 0.7152 | OK |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_2_init1` | last | False | last | last |  |  | PENDING |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_4_init0` | last | True | last | last |  | 0.7406 | OK |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_4_init1` | last | False | last | running:no best_ckpt | last |  | PENDING |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_cont4to6_init0` | last | True | last | last |  | 0.7748 | OK |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_cont4to6_init1` | advbest | False | advbest | pending:no best_ckpt | advbest |  | PENDING |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_cont4to8_init0` | last | True | last | last |  | 0.7886 | OK |
| convnext_base v1_madry_l2 | `convnext_base_v1_l2_cont4to8_init1` | last | False | last | pending:no best_ckpt | last |  | PENDING |
| convnext_base gradnorm_l1 | `convnext_base_gradnorm_l1_1_init0` | best | True | best | best |  | 0.6125 | OK |
| convnext_base gradnorm_l1 | `convnext_base_gradnorm_l1_1_init1` | last | False | last | pending:no best_ckpt | last |  | PENDING |
| convnext_base gradnorm_l1 | `convnext_base_gradnorm_l1_2_init0` | last | True | last | last |  | 0.6106 | FLAG |
| convnext_base gradnorm_l1 | `convnext_base_gradnorm_l1_2_init1` | advbest | False | advbest | running:no best_ckpt | advbest |  | PENDING |
| vit_b_cvst madry_l1 | `vit_b_cvst_l1_1_init1` | advbest | True | advbest |  | advbest | 0.6128 | OK |
| vit_b_cvst madry_l1 | `vit_b_cvst_l1_2_init1` | last | True | last |  | last | 0.6793 | OK |
| vit_b_cvst madry_l1 | `vit_b_cvst_l1_4_init1` | last | True | last |  | last | 0.7211 | OK |
| vit_b_cvst madry_l1 | `vit_b_cvst_l1_cont4to6_init1` | last | True | last |  | last | 0.7201 | OK |
| vit_b_cvst madry_l1 | `vit_b_cvst_l1_cont4to8_init1` | last | True | last |  | last | 0.7369 | OK |
| vit_b_cvst madry_l2 | `vit_b_cvst_l2_1_init1` | advbest | True | advbest |  | advbest | 0.7250 | OK |
| vit_b_cvst madry_l2 | `vit_b_cvst_l2_2_init1` | advbest | True | advbest |  | advbest | 0.7764 | OK |
| vit_b_cvst madry_l2 | `vit_b_cvst_l2_4_init1` | advbest | True | advbest |  | advbest | 0.8131 | OK |
| vit_b_cvst madry_l2 | `vit_b_cvst_l2_cont4to6_init1` | last | True | last |  | last | 0.8311 | OK |
| vit_b_cvst madry_l2 | `vit_b_cvst_l2_cont4to8_init1` | last | True | last |  | last | 0.8406 | OK |
| vit_b_cvst madry_linf | `vit_b_cvst_linf_1_init1` | best | True | best |  | best | 0.6663 | OK |
| vit_b_cvst madry_linf | `vit_b_cvst_linf_2_init1` | best | True | best |  | best | 0.7288 | OK |
| vit_b_cvst madry_linf | `vit_b_cvst_linf_4_init1` | advbest | True | advbest |  | advbest | 0.7914 | OK |
| vit_b_cvst madry_linf | `vit_b_cvst_linf_cont4to6_init1` | last | True | last |  | last | 0.7929 | OK |
| vit_b_cvst madry_linf | `vit_b_cvst_linf_cont4to8_init1` | advbest | True | advbest |  | advbest | 0.8213 | OK |
| swin_b madry_l1 | `swin_b_l1_1_init1` | advbest | True | advbest |  | advbest | 0.5867 | OK |
| swin_b madry_l1 | `swin_b_l1_2_init1` | last | True | last |  | last | 0.6580 | OK |
| swin_b madry_l1 | `swin_b_l1_4_init1` | advbest | True | advbest |  | advbest | 0.7060 | OK |
| swin_b madry_l1 | `swin_b_l1_cont4to6_init1` | last | True | last |  | last | 0.7231 | OK |
| swin_b madry_l1 | `swin_b_l1_cont4to8_init1` | last | True | last |  | last | 0.7385 | OK |
| swin_b madry_l2 | `swin_b_l2_1_init1` | last | True | last |  | last | 0.6852 | OK |
| swin_b madry_l2 | `swin_b_l2_2_init1` | advbest | True | advbest |  | advbest | 0.7287 | OK |
| swin_b madry_l2 | `swin_b_l2_4_init1` | last | True | last |  | last | 0.7978 | OK |
| swin_b madry_l2 | `swin_b_l2_cont4to6_init1` | last | True | last |  | last | 0.8181 | OK |
| swin_b madry_l2 | `swin_b_l2_cont4to8_init1` | best | True | best |  | best | 0.8519 | OK |
| swin_b madry_linf | `swin_b_linf_1_init1` | last | True | last |  | last | 0.6287 | OK |
| swin_b madry_linf | `swin_b_linf_2_init1` | best | True | best |  | best | 0.7135 | OK |
| swin_b madry_linf | `swin_b_linf_4_init1` | best | True | best |  | best | 0.7587 | OK |
| swin_b madry_linf | `swin_b_linf_cont4to6_init1` | last | True | last |  | last | 0.7677 | OK |
| swin_b madry_linf | `swin_b_linf_cont4to8_init1` | last | True | last |  | last | 0.7895 | OK |
