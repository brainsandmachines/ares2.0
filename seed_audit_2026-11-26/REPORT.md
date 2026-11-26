# Seed / reproducibility audit — 2026-11-26

Read-only audit. No code, CSV, DB, or campaign state was changed.

Files in this directory:

| file | content |
|---|---|
| `model_seeds.csv` | one row per curated model in `/mnt/data4t/models` (389): init seed, its source, every training seed, launch segments (`ep<start>:<seed>`), status flags, near-clone partners |
| `launches.csv` | every logged launch since 2026-04-12 (1,540, Botero + cluster run dirs, deduplicated): model, timestamp, seed, start epoch |
| `epoch0_fingerprint_clusters.txt` | models whose epoch-0 `(train_loss, eval_loss)` are bit-identical, meaning they started from the same weights and data order |

## 1. RNG sources

| RNG source | Seeded? | Where | Recorded? |
|---|---|---|---|
| Python `random`, NumPy, torch CPU (main process) | yes, `seed + rank` | `ares/utils/dist.py:83` ← `robust_training/adversarial_training.py:99` | seed in `log.txt` (every launch), `runtime_config.yaml` (last launch only), checkpoint `args.config.seed` (launch that saved it) |
| torch CUDA (TRADES random start, validation PGD random start) | yes. `torch.manual_seed` seeds all CUDA devices, so no separate `cuda.manual_seed_all` is needed | same | same |
| **Seed value itself** | `seed: null` default, so `torch.randint` draws a fresh value from OS entropy **on every launch, including every resume** | `adversarial_training.py:96`; `config.yaml:22` | see above |
| Before 2026-04-12 18:15 (commit `6556c230`) | **`seed: 0` fixed for every run** | `config.yaml` | `args.yaml` (holds the *last pre-June launch's* seed, so it is not trustworthy for the first launch) |
| RNG state on resume | **not saved/restored**; a new seed is drawn | — | — |
| cuDNN | **non-deterministic**: `deterministic=False`, `benchmark=True` | `adversarial_training.py:100-101` | no |
| `use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` | not set | — | — |
| Train shuffle (single GPU: `shuffle=True`) | implicitly, from the torch global RNG | `ares/utils/dataset.py:87` | — |
| DataLoader workers (RandomResizedCrop, flips, RandAugment, RandomErasing; Mixup/CutMix run in `collate_fn` inside workers) | implicitly: PyTorch seeds each worker's torch/random/numpy from `base_seed + worker_id`, drawn from the main RNG. No `worker_init_fn`/`generator`. Stream depends on `num_workers` | `dataset.py:84` | `num_workers` in config |
| `DistributedSampler` (DDP only; sjm tasks are 1 GPU so it's unused) | **`seed=0` default, ignores `cfg.seed`**; `set_epoch` is called | `dataset.py:66`, `adversarial_training.py:305` | — |
| Per-rank seeding | `seed + rank` | `dist.py:84` | rank in log |
| Madry PGD random init | n/a: `random_start=False` hard-coded | `ares/utils/train_loop.py:85,87` | — |
| TRADES random init | CUDA RNG (`torch.rand_like`), seeded; off for L1 | `train_loop.py:89`, `ares/utils/adv.py` | — |
| DVD age curve | `cfg.seed` | `ares/utils/dvd.py:84,95` | with the seed |
| V1 Gabor bank | `v1_gabor_seed` or `cfg.seed` fallback. Weights become an `nn.Parameter`, so they are saved and restored on resume (verified in a checkpoint) | `ares/utils/model.py:24`, `ares/model/v1_block.py:148` | in `state_dict` ✓ |
| AutoAttack in the sjm-eval lane / in-job sweep: **image subset** + APGD/Square seeds | `seed=int(cfg.seed or 0)`, i.e. the eval process's **fresh random seed** | `ares/utils/final_eval_helpers.py:445` → `data_analysis/autoattack_array_eval.py:391` | `autoattack_sweep_selection.json` (seed + indices) ✓ |
| `aa_sweep` / `autoattack_array_eval` standalone | `--seed 0` | sbatches | ✓ |
| `data_analysis/final_eval.py` AutoAttack | global `set_seed(0)`, but `AutoAttack(seed=None)` gives a `time.time()` seed per attack | `final_eval.py:404`, `ares/attack/autoattack/autoattack.py:92` | no |
| `data_analysis/autoattack_eval.py` | seed 0 + deterministic cuDNN | `:96-101` | — |

**What is persisted**
- `log.txt` (append mode) has a `Using seed: N` line for every launch since 2026-04-12. This is the only complete history.
- `runtime_config.yaml` is overwritten each launch. In 233/400 runs it differs from the first launch's seed.
- `hydra_config.yaml` is captured before resolution, so it always says `seed: null`.
- Checkpoints: `args.config` = full resolved config incl. the seed of the launch that saved it. Cross-check against the logs: so far every checkpoint's seed matches one of that model's logged launch seeds.
- Job DBs (`slurm_job_manager/jobs.sqlite`, frozen AIRCC DB): **no seed, no config, no command line.**
- Job CSVs: **no `seed` column.** `init` is in `METADATA_COLUMNS` (`slurm_job_manager/csv_spec.py:31`), so it is never forwarded to training.
- Nowhere: git commit, GPU model, torch/CUDA/cuDNN versions, determinism flags.

## 2. Did init0/1/2 get distinct seeds?

`init` was never forwarded, so any distinctness came from the seed default, not from `init`.

- **Since 2026-04-12 18:15: yes, by chance.** 1,540 logged launches, 0 reused seeds. All 111 sibling groups (same config, different `init`) have different init seeds. Every convnext_base / swin_b / ViT model has a unique epoch-0 fingerprint.
- **Before that: no.** Every run used `seed: 0`. That covers all **77 convnext_small models first launched before 2026-04-12** (init1–init5 alike, `status=INIT_SEED_0_SHARED` in `model_seeds.csv`). The `init4` runs launched 2026-04-10 are in this set, even though their `args.yaml` shows a random seed from a later relaunch.
  - **37** of them have a bit-identical epoch-0 `(train_loss, eval_loss)`. One cluster of 29 spans protocols: l1/l2/linf, eps 1–16, init1–init5. Siblings that don't match at epoch 0 diverged through code changes between launches or cuDNN non-determinism, but they still started from the same initial weights.
  - **8 sibling pairs are near-clones**, with identical train loss for N leading epochs:

    | pair | identical epochs | final top1 a / b | final adv-top1 a / b |
    |---|---|---|---|
    | l2_16 init1/init2 | 193 / 201 | 72.22 / 72.34 | 31.78 / 31.61 |
    | l1_16 init1/init2 | 186 / 201 | 70.95 / 70.94 | 50.61 / 50.65 |
    | l2trades_8 init1/init2 | 108 / 201 | 72.08 / 71.99 | 32.99 / 33.09 |
    | linf_16 init1/init2 | 98 / 202 | 73.87 / 74.23 | 39.79 / 40.36 |
    | linftrades_2 init1/init2 | 55 / 201 | 76.33 / 76.29 | 69.29 / 69.43 |
    | linf_4 init1/init5 | 41 / 210 | 73.84 / 73.77 | 48.69 / 48.81 |
    | l2_8 init1/init2 | 31 / 203 | 71.32 / 71.27 | 43.22 / 37.69 |
    | linf_2 init1/init2 | 12 / 208 | 75.46 / 75.38 | 59.41 / 60.06 |

    Any across-init variance computed from these convnext_small pairs is underestimated. The first two pairs are effectively n=1.

**No model has an unknown seed.** For seed-0-era models, epoch-0 seed = 0 (config default, and no launcher in the repo ever overrode it). Every later launch is logged.

**But "what seed produced model X" has a list answer for most models.** 182/389 models trained across more than one seed (median 4), because every requeue drew a new seed and the RNG state was not restored. `model_seeds.csv` gives the per-segment map (`ep0:<seed>;ep18:<seed>;…`). Even with all seeds known, these runs cannot be replayed bit-for-bit: resume reseeding plus non-deterministic cuDNN prevents it.

**Eval side.** 257 models' AutoAttack numbers use their own random 1,024-image subset, and 84 use the shared seed-0 subset. Each subset is recorded, but cross-model AA comparisons are not on identical images.

34 run dirs exist only on the cluster (in-flight `init4` runs not yet synced, V1 `convnext_small` sweeps, `vit_b_cvst_broken`). All are post-cutoff with logged random seeds; they're in `launches.csv`, not in `model_seeds.csv`.

## 3. Minimal patch plan (not applied)

Existing CSV rows have no seed and would keep exactly today's behaviour, so running campaigns are untouched. Everything that changes RNG behaviour is gated on an explicit seed.

**P1 — record (additive, safe to deploy any time)**
1. New `ares/utils/provenance.py: record_launch(...)`, called right after seeding (`adversarial_training.py:~99`, rank 0). It appends one JSON line to `<run_dir>/seed_history.jsonl` with: ts, seed, seed_source (`cli` / `drawn` / `resumed`), start_epoch, world_size, SLURM job/array ids, host, GPU name, git commit + dirty flag, torch/CUDA/cuDNN/timm versions, cuDNN flags, `sys.argv`.
2. Put `seed_history` + `provenance` into `checkpoint_metadata` (`adversarial_training.py:244`) so every `.pth.tar` carries the full history. It travels with rsync and HF uploads.
3. No DB schema change: the DB is operational state, and the CSV + run dir are the record.

**P2 — control (only rows that set it)**
1. Add `seed` to `OVERRIDE_COLUMNS` (`slurm_job_manager/csv_spec.py`). Empty cells emit nothing.
2. With an explicit seed, reseed at each epoch start with `derive(seed, epoch)`. A resumed run at epoch k then sees the same shuffle, augmentation and TRADES-init stream as an uninterrupted run. Without this, a fixed seed plus resume would replay epoch 0's stream at epoch k.
3. Guard in `seed.py`: reject new rows where same-group siblings with different `init` share a seed, or where `init` is set but `seed` is empty.
4. `DistributedSampler(dataset_train, seed=cfg.seed)` (DDP only).

**P3 — determinism (opt-in flag, default off)**
`reproducibility.deterministic=true` sets cuDNN deterministic, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (exported in the sbatch before CUDA init). This is slower (to be measured on ConvNeXt), and it is bitwise-reproducible only on the same GPU model + software stack. Intended for a reviewer-facing spot check: re-run the first N epochs of a model and match its `summary.csv`.

**P4 — eval**
1. `final_eval_helpers.py:445`: use a fixed `final_eval_aa_seed` (default 0) instead of `cfg.seed`, so new models are evaluated on the same subset as the 84 seed-0 ones. This changes the eval protocol for new evaluations; existing results stay as recorded.
2. `data_analysis/final_eval.py:404`: `AutoAttack(..., seed=0)`.
