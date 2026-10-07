# AIRCC copy audit — cluster vs QNAP vs `/mnt/data4t/models`

*2026-10-07 · read-only. Nothing was moved, copied, deleted or rsync'd. The sshfs mount (`~/slurm_mount`, ro) and the QNAP share were both up. No ssh was used.*

## TL;DR

The frozen AIRCC DB has **127** `finished` rows. **3** never wrote a checkpoint (`convnext_base_{l2_cont4to6,linf_cont4to6,linf_cont4to8}_init0`), so they exist nowhere. The other **124** break down like this:

| group | models | severity |
|---|---|---|
| **A. Cluster copy is a different (failed) training attempt; store dir is stale or mixed** | **5** | **high**: wrong AA grids, and 2 zoo links + 2 HF uploads point at collapsed weights |
| B. Cluster copy is missing keeper checkpoints (QNAP and store have them) | 10 | low: the AIRCC-side grids exist, but the store mixes two image selections |
| C. QNAP master is missing non-blessed keepers (cluster and store are the only copies) | 5 | low: provenance only |
| D. Minor: score bookkeeping | 2 | cosmetic |
| All copies agree (checkpoint content, json, CSV epoch and clean acc) | 102 | — |

Group A is the same failure as the `convnext_base_dvd_b_linf_2_init1` finding that started this audit, and it hits **5 models**:
`dvd_b_linf_2_init1`, `dvd_b_linf_1_init1`, `dvd_b_l1_1_init1`, `l2_4_init1` and `dvd_b_linf_2_init0` (all `convnext_base_*`). Each one's AIRCC run collapsed, was archived under `models_failed/` (`…__catastrophic_overfit_20260812`, `…__2026-08-14_aa1.37`, `…__gradient_collapse_20260806`) and was then rerun on AIRCC. The cluster still holds the **failed attempt**, and the aa_sweep Slurm lane has finished all **5 × 45 = 225 cells** on those failed checkpoints. The rerun's grid covers just 1 of 45 cells (the training-time eps_norm cell), with one exception: `dvd_b_l1_1_init1` best and advbest, which the Botero lane computed on the correct store checkpoints.

The most urgent problem is in the store. For **`dvd_b_l1_1_init1` and `dvd_b_linf_1_init1`**, the blessed `last.pth.tar` in `/mnt/data4t/models` is the **collapsed** checkpoint, which scores 1.17 and 0.10 robust. `models_for_experiments` links to it, the zoo manifest credits it with the rerun's 52.83 and 59.08, and it was exported to Hugging Face. The HF `metadata.json` records `size 1418062751` and the current store inode for both.

## How the copies were compared

- **Copies.** These are `cluster` (`~/slurm_mount/projects/ares/results/models/<name>`), `QNAP` (`/mnt/botero/aircc_archive/models/<name>`) and `store` (`/mnt/data4t/models/<arch>/<name>`). I also read `/mnt/botero/slurm_archive/models/<name>`, the route-1 mirror of the cluster. It matched the cluster byte-for-byte on every flagged file, so it is not shown separately.
- **Checkpoint identity.** I did not use mtime: 70 AIRCC `model_best` files share the 2026-08-10 bulk-rewrite mtime. I also did not hash 3 × 1.4 GB per model over sshfs. Instead, each checkpoint got two cheap reads. The first is a fingerprint built from the **CRC-32 of every tensor entry in the torch zip central directory**, which changes with the weights. The second is the `epoch` and `metric` fields, read from `data.pkl` without loading any tensors. Size alone is not enough: in `dvd_b_linf_2_init0` and `l2_4_init1` the failed attempt and the rerun have identical sizes.
- **Which copy is authoritative.** I treated the QNAP copy as authoritative where it holds the checkpoint. In all 124 models its `autoattack_eps_norm_scores.json` value for the DB's `best_checkpoint` equals the DB `best_score`. The one exception is `v1_noise_init0`, which is benign (group D). For the 3 failed attempts whose `models_failed/` archive still holds checkpoints, the **cluster checkpoints are byte-identical to the archived failed attempt**, matching on fingerprint for all three kinds.
- **CSV vs. checkpoint.** Every `autoattack_sweep_results{,_last,_advbest}.csv` in every copy was grouped by (epoch, clean acc). A group counts as mismatched when its epoch differs from the checkpoint beside it, or when its clean acc points at another variant of that kind held by a different copy. Clean acc and the checkpoint's `metric` differ by 3–5 points for several contepoch and pgd5 models. That gap is identical in all copies, so it is not flagged.
- I also compared the `autoattack_sweep_selection.json` index sets, because rows are only comparable on the same 1024 images. They diverge in exactly 10 models: the 5 of group A and the first 5 of group B. They agree in the other 114.

## Group A: different training attempt on the cluster, stale or mixed store

| model | DB blessed (score) | cluster | store `/mnt/data4t/models` | zoo link → | AA cells on the wrong checkpoint | suggested fix (not executed) |
|---|---|---|---|---|---|---|
| `convnext_base_dvd_b_linf_2_init1` | `model_best` **47.95** | all 3 ckpts = failed attempt (best ep160, last ep199, advbest ep151); json 0.00; 45/45 cells, all robust 0 | **mixed**: `model_best` + `model_best_adv` = rerun ✅; `last.pth.tar` = failed ❌; json = rerun; all 3 sweep CSVs = cluster's failed-attempt rows (ep160/199/151, clean ≈78, robust 0); selection = QNAP's | `madry/linf/…dvd_b_linf_2_init1` → `model_best.pth.tar` (**correct weights**, wrong CSVs beside it) | cluster 45; store 45 (copied) | QNAP is authoritative. Store: replace `last.pth.tar` with the QNAP one and restore the QNAP's 3 CSVs and `autoattack_sweep_selection.json`. Cluster: retire the failed dir and stage the QNAP keepers. **Recompute all 45 cells** (best, last, advbest) on the rerun. |
| `convnext_base_dvd_b_linf_1_init1` | `last` **59.08** | all 3 = failed attempt (best/advbest ep191, last ep199); json 0.10; 45/45 cells | **entirely the failed attempt**: all 3 ckpts ❌ and all 3 CSVs ❌, but json = rerun (claims 59.08) and selection = QNAP's | `madry/linf/…dvd_b_linf_1_init1` → `last.pth.tar` = **collapsed checkpoint (≈0.10 robust)**; also on HF | cluster 45; store 45 | QNAP is authoritative (rerun: best/advbest ep143, last ep199). Store: replace all 3 keepers and the CSVs with QNAP's. Cluster: as above. **Recompute all 45.** Then rebuild the zoo and **re-export to HF**. |
| `convnext_base_dvd_b_l1_1_init1` | `last` **52.83** | all 3 = failed attempt (best ep172, last ep199, advbest ep171); json 1.17; 45/45 cells | **mixed**: `model_best` (ep197) and `model_best_adv` (ep199) = rerun ✅, with **valid 15-cell CSVs** (Botero lane, Sep 10–14); `last.pth.tar` = failed ❌ and its CSV = cluster's failed rows; json = rerun | `madry/l1/…dvd_b_l1_1_init1` → `last.pth.tar` = **collapsed checkpoint (≈1.17 robust)**; also on HF | cluster 45; store 15 (`last`) | QNAP is authoritative. Store: replace `last.pth.tar` and drop the `_last` CSV. **Keep** the store's best/advbest CSVs (the only valid full grids for this model). Cluster: as above. **Recompute 15 cells (last) on the store, or 45 on the cluster.** Then rebuild the zoo and **re-export to HF**. |
| `convnext_base_l2_4_init1` | `last` **35.84** | all 3 = failed attempt (best ep141, last ep149, advbest ep143); json 0.00; 45/45 cells | **mixed**: `last` = rerun ✅; `model_best` (ep141) and `model_best_adv` (ep143) = failed ❌; CSVs = QNAP's 1-row eps_norm files, so the **best/advbest CSV rows say ep37 while the ckpts beside them are ep141/143** | `madry/l2/…l2_4_init1` → `last.pth.tar` (**correct**) | cluster 45 | QNAP is authoritative (rerun best = advbest = ep37). Store: replace `model_best` and `model_best_adv` with QNAP's. Cluster: as above. **Recompute all 45.** |
| `convnext_base_dvd_b_linf_2_init0` | `last` **0.10** | all 3 = first attempt (best ep161, last ep199, advbest ep169); 45/45 cells | ckpts = rerun ✅ (best ep198, last ep199, advbest ep174); **all 3 CSVs = cluster's first-attempt rows** (ep161/199/169, clean ≈79) | `madry/linf/…dvd_b_linf_2_init0` → `last.pth.tar` (**correct**) | cluster 45; store 45 | Both attempts collapsed (DB 0.10), so the science barely changes. Still: QNAP is authoritative; restore the QNAP CSVs and selection in the store, fix the cluster dir, **recompute 45** (or exclude the model from the sweep as a known collapse). |

Per-kind detail follows. Each cell shows size and mtime, the json score, the training norm/eps row of the CSV, and how many of the 15 grid cells that CSV covers.

#### `convnext_base_dvd_b_linf_2_init1`

Threat model linf eps 2. DB `best_checkpoint` = `model_best.pth.tar`, `best_score` = **47.95**.

| kind | QNAP (authoritative) | cluster | store |
|---|---|---|---|
| best | ckpt ep199, 1418062943, mtime 2026-08-27<br>json 47.95<br>eps_norm row: ep199, clean 68.8, robust **47.95**, 2026-08-27T08:33 (1/15 cells) | ckpt ep160, 1418062751, mtime 2026-08-08 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep160, clean 77.9, robust **0.00**, 2026-08-11T05:39 (15/15 cells) | ckpt ep199, 1418062943, mtime 2026-08-27 ✅ same<br>json 47.95<br>eps_norm row: ep160, clean 77.9, robust **0.00**, 2026-08-11T05:39 (15/15 cells) |
| last | ckpt ep199, 1418062943, mtime 2026-08-27<br>json 47.95<br>eps_norm row: ep199, clean 68.8, robust **47.95**, 2026-08-27T09:49 (1/15 cells) | ckpt ep199, 1418062751, mtime 2026-08-11 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep199, clean 78.6, robust **0.00**, 2026-08-11T05:41 (15/15 cells) | ckpt ep199, 1418062751, mtime 2026-08-11 ❌ **failed attempt**<br>json 47.95<br>eps_norm row: ep199, clean 78.6, robust **0.00**, 2026-08-11T05:41 (15/15 cells) |
| advbest | ckpt ep199, 1418082569, mtime 2026-08-27<br>json 47.95<br>eps_norm row: ep199, clean 68.8, robust **47.95**, 2026-08-27T11:06 (1/15 cells) | ckpt ep151, 1418082313, mtime 2026-08-08 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep151, clean 78.0, robust **0.00**, 2026-08-11T05:42 (15/15 cells) | ckpt ep199, 1418082569, mtime 2026-08-27 ✅ same<br>json 47.95<br>eps_norm row: ep151, clean 78.0, robust **0.00**, 2026-08-11T05:42 (15/15 cells) |

#### `convnext_base_dvd_b_linf_1_init1`

Threat model linf eps 1. DB `best_checkpoint` = `last.pth.tar`, `best_score` = **59.08**.

| kind | QNAP (authoritative) | cluster | store |
|---|---|---|---|
| best | ckpt ep143, 1418062943, mtime 2026-08-22<br>json 0.59<br>eps_norm row: ep143, clean 71.5, robust **0.59**, 2026-08-26T01:03 (1/15 cells) | ckpt ep191, 1418062751, mtime 2026-08-10 ❌ **failed attempt**<br>json 0.10<br>eps_norm row: ep191, clean 76.9, robust **0.10**, 2026-08-10T03:54 (15/15 cells) | ckpt ep191, 1418062751, mtime 2026-08-10 ❌ **failed attempt**<br>json 0.59<br>eps_norm row: ep191, clean 76.9, robust **0.10**, 2026-08-10T03:54 (15/15 cells) |
| last | ckpt ep199, 1418062943, mtime 2026-08-26<br>json 59.08<br>eps_norm row: ep199, clean 69.7, robust **59.08**, 2026-08-26T01:19 (1/15 cells) | ckpt ep199, 1418062751, mtime 2026-08-10 ❌ **failed attempt**<br>json 0.10<br>eps_norm row: ep199, clean 76.6, robust **0.10**, 2026-08-10T04:01 (15/15 cells) | ckpt ep199, 1418062751, mtime 2026-08-10 ❌ **failed attempt**<br>json 59.08<br>eps_norm row: ep199, clean 76.6, robust **0.10**, 2026-08-10T04:01 (15/15 cells) |
| advbest | ckpt ep143, 1418082569, mtime 2026-08-22<br>json 0.59<br>eps_norm row: ep143, clean 71.5, robust **0.59**, 2026-08-26T02:41 (1/15 cells) | ckpt ep191, 1418082313, mtime 2026-08-09 ❌ **failed attempt**<br>json 0.10<br>eps_norm row: ep191, clean 76.9, robust **0.10**, 2026-08-10T04:06 (15/15 cells) | ckpt ep191, 1418082313, mtime 2026-08-09 ❌ **failed attempt**<br>json 0.59<br>eps_norm row: ep191, clean 76.9, robust **0.10**, 2026-08-10T04:06 (15/15 cells) |

#### `convnext_base_dvd_b_l1_1_init1`

Threat model l1 eps 1. DB `best_checkpoint` = `last.pth.tar`, `best_score` = **52.83**.

| kind | QNAP (authoritative) | cluster | store |
|---|---|---|---|
| best | ckpt ep197, 1418062943, mtime 2026-08-27<br>json 52.25<br>eps_norm row: ep197, clean 67.4, robust **52.25**, 2026-08-27T04:25 (1/15 cells) | ckpt ep172, 1418062751, mtime 2026-08-10 ❌ **failed attempt**<br>json 1.37<br>eps_norm row: ep172, clean 77.9, robust **1.37**, 2026-08-09T18:46 (15/15 cells) | ckpt ep197, 1418062943, mtime 2026-08-27 ✅ same<br>json 52.25<br>eps_norm row: ep197, clean 67.4, robust **52.25**, 2026-08-27T04:25 (15/15 cells) |
| last | ckpt ep199, 1418062943, mtime 2026-08-27<br>json 52.83<br>eps_norm row: ep199, clean 67.6, robust **52.83**, 2026-08-27T06:48 (1/15 cells) | ckpt ep199, 1418062751, mtime 2026-08-10 ❌ **failed attempt**<br>json 1.17<br>eps_norm row: ep199, clean 77.6, robust **1.17**, 2026-08-09T19:45 (15/15 cells) | ckpt ep199, 1418062751, mtime 2026-08-10 ❌ **failed attempt**<br>json 52.83<br>eps_norm row: ep199, clean 77.6, robust **1.17**, 2026-08-09T19:45 (15/15 cells) |
| advbest | ckpt ep199, 1418082505, mtime 2026-08-27<br>json 52.83<br>eps_norm row: ep199, clean 67.6, robust **52.83**, 2026-08-27T09:09 (1/15 cells) | ckpt ep171, 1418082313, mtime 2026-08-08 ❌ **failed attempt**<br>json 1.17<br>eps_norm row: ep171, clean 78.0, robust **1.17**, 2026-08-09T20:43 (15/15 cells) | ckpt ep199, 1418082505, mtime 2026-08-27 ✅ same<br>json 52.83<br>eps_norm row: ep199, clean 67.6, robust **52.83**, 2026-08-27T09:09 (15/15 cells) |

#### `convnext_base_l2_4_init1`

Threat model l2 eps 4. DB `best_checkpoint` = `last.pth.tar`, `best_score` = **35.84**.

| kind | QNAP (authoritative) | cluster | store |
|---|---|---|---|
| best | ckpt ep37, 1418062751, mtime 2026-08-08<br>json 0.00<br>eps_norm row: ep37, clean 75.3, robust **0.00**, 2026-08-15T15:48 (1/15 cells) | ckpt ep141, 1418062751, mtime 2026-08-03 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep141, clean 76.8, robust **0.00**, 2026-08-03T09:07 (15/15 cells) | ckpt ep141, 1418062751, mtime 2026-08-03 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep37, clean 75.3, robust **0.00**, 2026-08-15T15:48 (1/15 cells) |
| last | ckpt ep149, 1418062751, mtime 2026-08-15<br>json 35.84<br>eps_norm row: ep149, clean 67.5, robust **35.84**, 2026-08-15T15:50 (1/15 cells) | ckpt ep149, 1418062751, mtime 2026-08-03 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep149, clean 76.9, robust **0.00**, 2026-08-03T09:08 (15/15 cells) | ckpt ep149, 1418062751, mtime 2026-08-15 ✅ same<br>json 35.84<br>eps_norm row: ep149, clean 67.5, robust **35.84**, 2026-08-15T15:50 (1/15 cells) |
| advbest | ckpt ep37, 1418082313, mtime 2026-08-08<br>json 0.00<br>eps_norm row: ep37, clean 75.3, robust **0.00**, 2026-08-15T17:18 (1/15 cells) | ckpt ep143, 1418082313, mtime 2026-08-03 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep143, clean 76.7, robust **0.00**, 2026-08-03T09:10 (15/15 cells) | ckpt ep143, 1418082313, mtime 2026-08-03 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep37, clean 75.3, robust **0.00**, 2026-08-15T17:18 (1/15 cells) |

#### `convnext_base_dvd_b_linf_2_init0`

Threat model linf eps 2. DB `best_checkpoint` = `last.pth.tar`, `best_score` = **0.10**.

| kind | QNAP (authoritative) | cluster | store |
|---|---|---|---|
| best | ckpt ep198, 1418062751, mtime 2026-08-18<br>json 0.00<br>eps_norm row: ep198, clean 72.2, robust **0.00**, 2026-08-18T20:08 (1/15 cells) | ckpt ep161, 1418062751, mtime 2026-08-01 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep161, clean 79.7, robust **0.00**, 2026-08-03T22:04 (15/15 cells) | ckpt ep198, 1418062751, mtime 2026-08-18 ✅ same<br>json 0.00<br>eps_norm row: ep161, clean 79.7, robust **0.00**, 2026-08-03T22:04 (15/15 cells) |
| last | ckpt ep199, 1418062751, mtime 2026-08-18<br>json 0.10<br>eps_norm row: ep199, clean 72.9, robust **0.10**, 2026-08-18T20:10 (1/15 cells) | ckpt ep199, 1418062751, mtime 2026-08-03 ❌ **failed attempt**<br>json 0.10<br>eps_norm row: ep199, clean 78.6, robust **0.10**, 2026-08-03T22:06 (15/15 cells) | ckpt ep199, 1418062751, mtime 2026-08-18 ✅ same<br>json 0.10<br>eps_norm row: ep199, clean 78.6, robust **0.10**, 2026-08-03T22:06 (15/15 cells) |
| advbest | ckpt ep174, 1418082377, mtime 2026-08-17<br>json 0.00<br>eps_norm row: ep174, clean 72.9, robust **0.00**, 2026-08-18T20:12 (1/15 cells) | ckpt ep169, 1418082313, mtime 2026-08-02 ❌ **failed attempt**<br>json 0.00<br>eps_norm row: ep169, clean 79.5, robust **0.00**, 2026-08-03T22:08 (15/15 cells) | ckpt ep174, 1418082377, mtime 2026-08-17 ✅ same<br>json 0.00<br>eps_norm row: ep169, clean 79.5, robust **0.00**, 2026-08-03T22:08 (15/15 cells) |

### How group A happened

1. **Step 2 merge (`02_merge_decisions.csv`) compared snapshots taken mid-rerun.** The frozen `/mnt/data4t/aircc_archive` copies were taken while the AIRCC reruns were still training: `last` at ep6 (`dvd_b_l1_1_init1`), ep22 (`dvd_b_linf_1_init1`), ep151 (`dvd_b_linf_2_init1`), and `best`/`advbest` at ep37 (`l2_4_init1`). `winner_by=epoch` therefore chose the slurm-archive copy, which is the failed attempt, as the keeper that Step 3 hardlinked into `models/` (those files still have `nlink == 2`).
2. **Backfill never repaired equal-epoch reruns from the AIRCC root.** Once the reruns finished, `backfill` replaced a store checkpoint only when the QNAP-AIRCC copy had a **strictly higher epoch**. Two things follow. First, every `last.pth.tar` (ep199 against ep199, ep149 against ep149) stayed stale. Second, `dvd_b_linf_1_init1`'s rerun best/advbest (ep143) lost to the failed attempt's ep191, and `l2_4_init1`'s rerun best/advbest (ep37) lost to ep141/143. The `qnap-slurm-rerun` exception added for exactly this case covers only the `qnap-slurm` root and only rows the **sjm** DB has as finished, and none of these 5 has an sjm row. That rule explains the mixed dirs exactly. For example, `dvd_b_linf_2_init1` got the rerun `best` and `advbest` (199 > 160 and 199 > 151), but not `last` (199 = 199).
3. **Weekly route 2 still re-imports the cluster's sweep CSVs.** For non-checkpoint files, `backfill.plan` (`--roots qnap-slurm`) pulls the newest copy from `/mnt/botero/slurm_archive`, which mirrors the cluster. The cluster's 45-cell CSVs, computed on the failed checkpoints, are newer and fuller than the QNAP-AIRCC 1-row files, so they overwrite the store's CSVs every Monday. **This will keep happening** after a manual fix of the store until the cluster dir itself is corrected. The json is not affected because the cluster's json is *older*, which is why the store json says 47.95 or 59.08 while the CSVs beside it say 0.
4. **The aa_sweep Slurm lane did what it was designed to do.** The model has a cluster dir, so it belongs to the Slurm lane, and that lane evaluates the cluster's own copy. None of these 5 has a pending or failed unit in `aa_queue.sqlite`, because their cluster CSVs are complete. `dvd_b_l1_1_init1` *also* has 3 finished Botero-lane rows (enqueued 2026-09-10, before the 09-28 cutover). Its `last` unit (2026-09-20) logged `Skipping complete matching sweep CSV`, because by then the store's `_last` CSV had been replaced by the cluster's. That CSV matched the equally stale store `last.pth.tar` at ep199, so nothing was recomputed.

## Group B: cluster dir is missing keeper checkpoints

On all of these, the QNAP and store hold the missing checkpoints, and their fingerprints match wherever the cluster has a copy too. The full AIRCC-side grid (18 rows: 15 grid cells plus eps 12) exists in the QNAP for every missing kind. Nothing here is computed on a wrong checkpoint.

| model | missing on cluster | cluster CSV for that kind | side effect in the store |
|---|---|---|---|
| `convnext_base_baseline_init0` (also sjm-finished) | `last` | none | `best` CSV and `selection.json` replaced by the cluster's 2026-09-06 re-sweep (different 1024 images, clean 82.8 against 81.5 on the same ep98 ckpt); `last` CSV is still on the QNAP selection |
| `convnext_base_baseline_init1` (also sjm-finished) | `last` | none | same pattern (`best` clean 83.8 against 82.3) |
| `convnext_base_l1_4_init0` | `best`, `last` | none | `advbest` CSV and selection replaced by the cluster's 09-28 re-sweep; best/last on the QNAP selection |
| `convnext_base_l2_4_init0` | `last`, `advbest` | none | `best` CSV and selection replaced by the cluster's 09-28 re-sweep |
| `convnext_base_linf_4_init0` | `best`, `advbest` | none | `last` CSV and selection replaced by the cluster's 09-28 re-sweep |
| `convnext_base_dvd_b_l1trades_4_init0` | `best` | 18 rows, ep165 = QNAP ckpt ✅ | none |
| `convnext_base_dvd_b_l2_1_init0` | `best`, `last` | 18 rows each, ep155/ep199 = QNAP ✅ | none |
| `convnext_base_dvd_b_linftrades_1_init0` | `best` | 18 rows, ep161 = QNAP ✅ | none |
| `convnext_base_linftrades_2_init0` | `best` | 18 rows, ep100 = QNAP ✅ | none |
| `convnext_base_linftrades_4_init0` | `best` | 18 rows, ep93 = QNAP ✅ | none |

**Suggested fix:** no recompute. For the first five, a store dir now pairs a `selection.json` with CSVs that were computed on another selection. That only matters if a future Botero-lane top-up adds cells to those CSVs. If cross-kind comparability within one dir matters, restore the QNAP CSVs and selection for those kinds in the store. The cluster's re-sweeps are valid numbers on the same weights, just on different images. Optionally, copy the missing keepers to the cluster so the cluster dir is whole. The Slurm lane never needs them, because a kind with no checkpoint is never planned there.

## Group C: QNAP master is missing non-blessed keepers

These are the AIRCC `cont*_pgd5` reruns that `blessing_overrides.csv` publishes under the `cont*_init0` names. In each one the **blessed kind is present on the QNAP and identical everywhere**, and the other keepers exist only on the cluster and in the store (as hardlinks from the frozen `/mnt/data4t/aircc_archive`). Their CSVs agree between the cluster and the store.

| model | QNAP lacks | blessed kind (QNAP = cluster = store) |
|---|---|---|
| `convnext_base_l1_cont4to6_lr1e4_pgd5_init0` | `model_best`, `model_best_adv` | `last` ep39 |
| `convnext_base_l1_cont4to8_pgd5_init0` | `model_best`, `model_best_adv` | `last` ep39 |
| `convnext_base_l2_cont4to6_lr1e4_pgd5_init0` | `model_best`, `model_best_adv` | `last` ep39 |
| `convnext_base_linf_cont4to6_pgd5_init0` | `model_best`, `last` | `model_best_adv` ep19 |
| `convnext_base_linf_cont4to8_pgd5_init0` | `model_best`, `model_best_adv` | `last` ep39 |

**Suggested fix:** none required. Just note that the "QNAP is the master" assumption does not hold for these 10 files. If `/mnt/data4t/aircc_archive` is ever staged for deletion, check these first: `nlink == 2` today means the archive and the store copy share an inode, not that a third copy exists.

## Group D: minor

- `convnext_base_l1_4_init1`: the cluster has no `autoattack_eps_norm_scores.json`. It recomputed the l1/eps4 advbest cell at batch 32 (33.20 against 33.40 at batch 128, same ckpt, same images), and that row reached the store, so the store json (33.40) and the store CSV (33.20) differ by 2 images. Cosmetic.
- `convnext_base_v1_noise_init0`: DB `best_score` 75.39 is clean accuracy (a noise-only model), while the json holds AA 19.82. This holds in all 3 copies, so it is not a copy disagreement.

## `models_for_experiments` links into flagged dirs

| zoo path | target | target is |
|---|---|---|
| `convnext_base/madry/l1/convnext_base_dvd_b_l1_1_init1.pth.tar` | `…/convnext_base_dvd_b_l1_1_init1/last.pth.tar` | ❌ **failed attempt (collapsed, ≈1.2 robust)**; manifest says 52.83; exported to HF |
| `convnext_base/madry/linf/convnext_base_dvd_b_linf_1_init1.pth.tar` | `…/convnext_base_dvd_b_linf_1_init1/last.pth.tar` | ❌ **failed attempt (collapsed, ≈0.1 robust)**; manifest says 59.08; exported to HF |
| `convnext_base/madry/linf/convnext_base_dvd_b_linf_2_init1.pth.tar` | `…/model_best.pth.tar` | ✅ rerun (dir is mixed: `last` and the CSVs are wrong) |
| `convnext_base/madry/l2/convnext_base_l2_4_init1.pth.tar` | `…/last.pth.tar` | ✅ rerun (dir is mixed: `model_best`/`model_best_adv` are wrong) |
| `convnext_base/madry/linf/convnext_base_dvd_b_linf_2_init0.pth.tar` | `…/last.pth.tar` | ✅ rerun (CSVs are wrong) |
| group B (`baseline_init{0,1}`, `l1_4_init0`, `l2_4_init0`, `linf_4_init0`) | blessed kind | ✅ correct weights; only the CSV/selection mix noted above |
| group C (5 `cont*_init0` names) | blessed kind | ✅ correct |

Any `epsilon_bounded_contstim` results that loaded the two ❌ links measured a collapsed model in place of the 52.8 / 59.1 robust rerun.

## Suggested order of operations

> **Step 1 done (2026-10-07 11:54–12:01).** On the cluster, each failed-attempt dir was moved to
> `results/models_failed/<name>__<reason>_slurmcopy`, following the existing convention. The QNAP
> rerun's 3 keepers and 9 metadata files (json, 3 one-row CSVs, selection, hydra/runtime config,
> summary, log) were then rsync'd into a fresh `results/models/<name>`; `periodic/` was not copied.
> Verification: every keeper's size and tensor-CRC fingerprint match the QNAP, every metadata file is
> byte-identical, and the census reports 14/15 cells missing per kind. A `submit --dry-run` would feed
> 3 units per model at the next 21:30 run.
>
> **Step 2 done (2026-10-07 12:13).** In `/mnt/data4t/models/convnext_base/<model>`, 25 files that
> differed from the QNAP rerun were replaced by real copies of it (written to a temp file, then renamed,
> never through a hardlink): 7 keepers, 11 sweep CSVs, 7 config/summary files. 14 failed-attempt sweep
> logs and plots were staged with no replacement. The 39 displaced files are in
> `/mnt/data4t/pending_deletion/2026-10-07_aircc_rerun_fix/`, with a `MANIFEST.csv` and `README.md`.
> `dvd_b_l1_1_init1`'s valid Botero-lane best/advbest CSVs were kept. A full re-audit shows store =
> cluster = QNAP for all 5, with no remaining issues. Every zoo link now resolves to rerun weights.
> **Caveat:** route 2 must not run before route 1 has refreshed `/mnt/botero/slurm_archive/models/<model>`.
> Run against today's archive, a backfill dry-run would pull back 4 failed checkpoints
> (`qnap-higher-epoch`: `dvd_b_linf_1_init1` best/advbest ep191, `l2_4_init1` best/advbest ep141/143)
> and 11 failed CSVs (`qnap-newer`). Steps 3–5 are not done.
>
> **Step 3 partly done (2026-10-07 12:21–12:22).** `zoo-apply` was a no-op: 0 links created or deleted,
> and `manifest.csv` was byte-identical to the planned one, because links resolve by path. `hf-apply --only`
> re-exported `dvd_b_l1_1_init1` and `dvd_b_linf_1_init1` (Hub commit `2b026baa7a`). **Side effect:** the
> regenerated model card and `manifest.csv` now list only 154 of the previous 161 models. Seven other
> models whose store source changed after the 2026-09-10 export dropped out of the index:
> `convnext_base_{dvd_b_l1trades_4_init1, dvd_b_l2trades_2_init1, dvd_b_linf_4_init1,
> dvd_b_linf_cont4to8_init1_contepoch, dvd_b_linftrades_2_init0, dvd_b_linftrades_4_init0}` and
> `vit_b_cvst_linftrades_4_init1`. Their old weights are still in the repo, and they are stale: each is an
> earlier checkpoint than the sjm-blessed one. `dvd_b_linf_4_init1` and `dvd_b_linftrades_{2,4}_init0`
> match names under the cluster's `models_failed/`. The current store source of all 7 equals the cluster's
> finished, sjm-blessed checkpoint, so re-exporting them is the fix.
>
> **Step 3 done (2026-10-07 12:29).** The 7 were re-exported (Hub commit `a1fac717a9`). The Hub index is back to
> 161 models, the same set as the 2026-09-10 export. For all 9 re-exported models, the Hub `metadata.json`
> source is the current store checkpoint (same file key), and its sha256 matches the manifest. 34
> never-published zoo models are still waiting for a full `hf-apply`, which has not been run.

## Remaining steps (not done)

1. ~~**Fix the cluster dirs of the 5 group-A models first.**~~ *(done, see above)* Otherwise route 2 re-imports the bad CSVs into the store every Monday. Move each failed-attempt dir aside on the cluster (e.g. into a `models_failed/`-style location, matching the AIRCC naming), then stage the QNAP rerun's `model_best`/`last`/`model_best_adv`, `autoattack_eps_norm_scores.json`, the 3 one-row CSVs and `autoattack_sweep_selection.json` into `results/models/<name>`. This is a one-off manual copy, which aa_sweep deliberately will not do. The next nightly `aa_sweep.submit` then sees 14 missing cells per kind and feeds 15 units. The alternative is to leave no cluster dir at all, which puts the model in the Botero lane (~25–40 h per checkpoint on the 4090, so ~15 checkpoint-runs ≈ 3 weeks).
2. ~~**Fix the store**~~ *(done, see above)* (route 2 will not do it: equal epoch, AIRCC root, no sjm row). Replace the stale keepers listed in the group A table with the QNAP files, as real copies. Move the displaced files into `/mnt/data4t/pending_deletion/<date>/` per `model_store` convention rather than deleting them. Restore the QNAP CSVs and selection, **except** `dvd_b_l1_1_init1`'s best/advbest CSVs, which are valid. Never write through the existing hardlinks (no `--inplace`).
3. *(done, see above)* ~~**Rebuild the zoo**~~ (`ms_run.sh zoo-apply`). The link targets do not change, but the manifest's `target sha256` will. **Re-export `dvd_b_l1_1_init1` and `dvd_b_linf_1_init1` to HF** (`ms_run.sh hf` should report them as *changed* through `file_key`).
4. *(done 2026-10-07: the sweep-CSV half, as `backfill._guard_results`, see `model_store/README.md`. The AIRCC equal-epoch half was dropped, because AIRCC is retired and route 2 reads `qnap-slurm` only.)* ~~**Consider a code fix**~~ so this cannot recur. Extend the equal-epoch rerun rule in `backfill.plan` to the `qnap-aircc` root, keyed on the frozen AIRCC DB row being `finished` and the bytes differing. Also consider making the metadata rule refuse a sweep CSV whose `epoch` column disagrees with the store checkpoint beside it.
5. Flag any downstream results (plots, papers, `epsilon_bounded_contstim` runs) that used the cluster or store AA grids for these 5 models, or the two ❌ zoo links.

---
*Raw data: one metadata pass over 127 models × 4 trees, ~50 s. Collector and analysis scripts are in this session's scratchpad. Re-running them is safe: they open every DB with `?immutable=1` and only `stat`/read files.*
