# Geirhos model-vs-human OOD evaluation (16 tasks)

This runs the 16 out-of-distribution tasks of [model-vs-human](https://github.com/bethgelab/model-vs-human)
on the HF-zoo checkpoints, and draws one page per task in the style of the shape-bias plot.
The 16 tasks are every model-vs-human dataset except cue-conflict. The style reference is the
last page of `data_analysis/plots/selected_models/selected_models_aa_shape_bias.pdf`.

**Cue-conflict / shape bias is excluded.** It is already scored by `data_analysis/shape_bias_selected.py`.
The registry refuses it (`select_tasks(["cue-conflict"])` raises), and nothing here reads or writes
`shape_bias_results_botero.csv` or the shape-bias PDF.

## Layout

| file | role |
|---|---|
| `tasks.py` | the 16 tasks. Each has its upstream id, conditions, `EXCLUDE_CONDITIONS` and expected trial / subject counts. |
| `models.py` | the 199 checkpoints of `epsilon_bounded_contstim/model_zoo/models_for_{adversarial,blackbox}_experiments.json`, grouped and ordered like the shape-bias report |
| `data.py` | read-only presence checks and the dataset class (local files only) |
| `metrics.py` | upstream metric definitions (see below) |
| `gpu.py` | sharing the GPU with another job: free-memory check, allocator cap, OOM back-off |
| `evaluate.py` | CLI: dry run, inference (per-trial CSVs) and scoring |
| `plot.py` | reference-style task pages, the multipage PDF and overview heatmaps |
| `analyze.py` | Madry vs TRADES, norm, eps and shape-bias connections → `analysis/` (`FINDINGS.md`, `tables/`, `plots/`) |
| `verify_extract.py` | size check + sha256 + safe extraction of hand-downloaded tarballs (stdlib only, no network) |
| `tests/` | unit and synthetic end-to-end tests; `test_gpu_coexist.py` is an opt-in live GPU check |
| `results/` | git-ignored (repo-wide `results/` rule). It holds `trials/<run_dir>__<kind>/<task>.csv`, `task_metrics`, `human_metrics`, `per_condition_accuracy`, `benchmark_summary_16tasks` (each as `.csv` + `.json`), `dry_run_status*`, `data_status`, `configs/run_*.json` and `logs/` |
| `plots/` | `NN_<task>.{pdf,png}`, `NN_<task>_values.{csv,json}` (the exact plotted numbers), `geirhos_ood_16_tasks.pdf` (or `geirhos_ood_INCOMPLETE_<k>of16.pdf`), `overview_heatmaps.pdf`, `plot_manifest.json` |

## Dependencies

Run everything in the `ares` conda env (torch, torchvision, PIL, pandas, numpy, matplotlib).
`modelvshuman` is **not** installed and never imported. Its decision mapping and metrics are
reimplemented in `metrics.py`, with a citation per function. The 16-class mapping is
`data_analysis/shape_bias/mapping.py`, the same one the shape-bias run used.

## Models

There are 199 checkpoints: 133 experiment picks plus their 66 other-init "black-box" siblings.
Each checkpoint's family and position come from `data_analysis.selected_models.resolve()`, so the
bars follow the shape-bias page.

Differences from that page:
- **Added:** the 30 vit_b_cvst / swin_b TRADES checkpoints, as `<arch> trades_<norm>` families
  after the arch's madry families.
- **Left out:** the 5 failed dvd_b other-init runs that `model_store/hf_excludes.csv` holds back,
  which is why neither JSON lists them.

Every checkpoint is loaded with `data_analysis.autoattack_array_eval.load_model`: EMA weights,
with normalisation inside the wrapper.

## Data (local only)

Images are read from `--data-root` or `$GEIRHOS_OOD_ROOT`. There is no default, and
`$MODELVSHUMANDIR` is ignored on purpose.
```
<root>/datasets/<task>/dnn/session-1/*.png     12 distortions, sketch, stylized
<root>/datasets/<task>/<category>/*.png        edge, silhouette
<root>/raw-data/<task>/<task>_subject-XX_session_1.csv   human trials
```

On Botero the root is `/mnt/data/datasets/model-vs-human`. It was populated on 2026-10-27
with the user's explicit Phase 2 authorisation:
```bash
cd /mnt/data/datasets/model-vs-human/_downloads
for t in colour contrast high-pass low-pass phase-scrambling power-equalisation false-colour rotation \
         eidolonI eidolonII eidolonIII uniform-noise sketch stylized edge silhouette; do
  curl -sS -L --fail -o $t.tar.gz https://github.com/bethgelab/model-vs-human/releases/download/v0.1/$t.tar.gz
done
curl -sS https://api.github.com/repos/bethgelab/model-vs-human/releases/tags/v0.1 -o release_v0.1.json
python -I experiments/geirhos_ood/verify_extract.py /mnt/data/datasets/model-vs-human \
    /mnt/data/datasets/model-vs-human/_downloads/release_v0.1.json colour contrast ...   # the 16 tasks
git clone --depth 1 https://github.com/bethgelab/model-vs-human.git model-vs-human-src   # for raw-data/ only
ln -s model-vs-human-src/raw-data raw-data
```
`MANIFEST.json` at the root records the size and sha256 of each tarball and the clone commit
(79ed7cd). The release publishes no checksums, so each size was checked against the GitHub API. Image
counts match the upstream human sessions: 1280 / 1120 for the distortions, 800 for sketch and
stylized, 160 for edge and silhouette. The human subject counts match too: 4, 7, 5 and 10.

### No-download guarantee
- The package has no network code. `tests/test_pipeline.py::test_sources_have_no_downloader`
  greps for one.
- Missing data shows up as a status (`missing_data`, `no_data_root`, `bad_layout`). It is never
  fetched and never scored as 0. Tests patch `socket.connect`, `urlopen` and `requests` to raise,
  then run the dry run and a full run on an empty root.
- `verify_extract.py` only reads tarballs that are already on disk.

## Commands
```bash
python -m experiments.geirhos_ood.evaluate --dry-run --data-root /mnt/data/datasets/model-vs-human
python -m experiments.geirhos_ood.evaluate --data-root /mnt/data/datasets/model-vs-human \
    --gpu-mem-budget-gb 6 --gpu-need-gb 3.5 --gpu-margin-gb 3          # resumable; --force re-runs
python -m experiments.geirhos_ood.evaluate --score-only --data-root /mnt/data/datasets/model-vs-human
python -m experiments.geirhos_ood.plot
python -m experiments.geirhos_ood.analyze      # after plot; reads results/ + the zoo JSONs + shape_bias_results_botero.csv
python -m pytest experiments/geirhos_ood/tests -q
GEIRHOS_GPU_COEXIST=1 GEIRHOS_OTHER_PID=<pid> python -m pytest experiments/geirhos_ood/tests/test_gpu_coexist.py -s
```
`--tasks` / `--models` select subsets. Models can be named by zoo key, run dir or `<run_dir>__<kind>`.
Each trial file is written atomically, and a file that already holds every image is skipped.
Scoring always rebuilds the derived tables from the trial files.

## Preprocessing

The pipeline uses the same preprocessing as the shape-bias run: `Resize(256)` (bilinear) →
`CenterCrop(224)` → `ToTensor`, in fp32. Normalisation happens in the model wrapper. Upstream's
own PyTorch wrappers also use Resize 256 / CenterCrop 224 on these 224 px stimuli.

## Metrics (upstream definitions)
- **Decision:** softmax → *mean* probability over each of the 16 categories' ImageNet classes
  → argmax. This is upstream's `decision_mappings.py` (`np.mean`). The reference plot's black
  dot uses the *max* aggregation instead; it is kept as `ood_accuracy_max`.
- **OOD accuracy:** the fraction of correct trials, pooled over every trial of the task's
  scored conditions. Scored conditions are the upstream conditions minus `EXCLUDE_CONDITIONS`:
  the undistorted control, plus the levels where mean human accuracy is below 0.2. Humans are
  pooled over all subjects' trials.
- **Accuracy difference:** (acc_model − acc_human)², with upstream's `np.square` norm.
- **Observed / error consistency:** trials are paired by `image_id`
  (`plotting_helper.get_short_imagename`). Observed consistency is the fraction of images both
  observers got right or both got wrong. Error consistency is Cohen's kappa with
  expected = p1·p2 + (1−p1)(1−p2).
- **Averaging:** compute per scored condition (or over the whole set for sketch, stylized,
  edge and silhouette), then average over conditions, then over human subjects. The human
  reference averages over unordered subject pairs.
- **Benchmark** (`benchmark_summary_16tasks`): the unweighted mean over the 16 tasks, only for
  models with all 16 measured. Upstream's leaderboard averages 17 datasets including
  cue-conflict, so this is **not** numerically the leaderboard figure.

## Plots

Each `NN_<task>` page reproduces the shape-bias report's last page
(`all_models_bars_page`):
- one bar per checkpoint, grouped by family with grey separators and family names on top
- the bar is mean-aggregation OOD accuracy: solid `#4C72B0` for the JSON pick, light
  `#A9BEDC` for the other init
- a black dot for max aggregation, and rotated value labels ("n/a" where a value is missing)
- bold tick labels for the picks, legend above the axes, y from 0 to 1.12

The one addition is a red dashed line for pooled human accuracy. Pages with missing models are
titled `INCOMPLETE`. The 16-page PDF is written only when every task has every model.
