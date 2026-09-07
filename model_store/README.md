# `model_store` — the curated model tree

Three trees, one purpose: give every trained checkpoint exactly one canonical home,
and one obvious answer to *"where is the best checkpoint for model X?"*.

| tree | what it is | written by |
|---|---|---|
| `/mnt/botero/{aircc,slurm}_archive` | the QNAP master archive. **Read-only from here.** | `slurm_job_manager/scripts/backup_slurm_models.sh` — append-only |
| `/mnt/data4t/models` | the curated working copy: every model, every keeper checkpoint, **no** `checkpoint-N` | `model_store.backfill` (QNAP copies) + `model_store.build_models` (hardlinks, local archives only) |
| `/mnt/data4t/models_for_experiments` | `<arch>/<protocol>/<norm>/<name>.pth.tar` symlinks to the blessed checkpoint | `model_store.build_experiments` |

`models_for_experiments` is the root `epsilon_bounded_contstim` reads
(`conf/machine/botero.yaml: models_zoo_path`).

## The two weekly routes

```
slurm results/models
    |  route 1  Sun 09:00  slurm_job_manager/scripts/backup_slurm_models.sh
    |           rsync over ssh, append-only, no checkpoint-N, skips running models
    v
/mnt/botero/slurm_archive          <- the master. Nothing local is authoritative.
    |  route 2  Mon 09:00  model_store/scripts/ms_weekly_sync.sh
    |           backfill --roots qnap-slurm: update-only, no checkpoint-N
    v
/mnt/data4t/models
    |           same script: build_experiments, relative symlinks
    v
/mnt/data4t/models_for_experiments  <- what experiments on Botero load
```

They are **independent on purpose**. Route 1 pulls ~0.9 TB over ssh and must not be
reported as failed because an index rebuild afterwards failed — that is exactly what
used to happen, and its zoo→QNAP leg could never succeed anyway (`/mnt/botero` is CIFS
with `nounix`, so creating a symlink there returns EIO). Route 2 must be re-runnable on
its own without re-pulling any of those bytes. They share only the archive, plus one
`flock` on route 1's lock so route 2 never indexes a half-finished pull.

**Experiments always read local copies.** Every link in `models_for_experiments` is
*relative* into `../models/` on `/mnt/data4t`; none points at the QNAP, and
`ms_weekly_sync.sh` fails if one ever does.

Two guards on route 2 worth knowing about, both about `--delete` in
`build_experiments.sync_into_place`:

- the zoo publishes only models with a DB-recorded `best_checkpoint`, and
  `census._read_db` returns `[]` for a DB path that does not exist — which is how
  `~/slurm_mount` looks when the sshfs has dropped. Route 2 therefore checks the sjm DB
  is readable and non-empty *before* rebuilding, and refuses otherwise.
- `--min-entries N` refuses a plan smaller than N. Route 2 passes 80% of the live entry
  count. A hand-run `ms_run.sh zoo-apply` passes none, so a deliberate large prune still
  works.

## Layout

```
/mnt/data4t/models/
├── convnext_base/convnext_base_l2_4_init1/{last,model_best,model_best_adv}.pth.tar
│                                          periodic/epoch_0090.pth.tar
│                                          log.txt summary.csv hydra_config.yaml
│                                          autoattack_sweep_results*.csv
│                                          _alt/data4t-aircc/log.txt   <- superseded metadata
├── convnext_small/convnext_small_l2trades_2_init1/...
├── swin_b/swin_b_l2_4_init1/...                    (was swin_b/l2_4_init1)
├── vit_b_cvst/vit_b_cvst_linf_cont4to8_init1/...
├── vit_m_cvst/...
├── _legacy/
│   ├── old_models/...            42 dirs
│   ├── vit_b_cvst_broken/...     14 dirs
│   └── unparsed/...              config and folder name disagree
└── _meta/                        archive-level provenance no model dir owns
    ├── aircc_archive/aircc_jobs_final_latest.sqlite   <- blesses 123 zoo entries
    ├── aircc_archive/pruned_20260827.manifest.jsonl
    └── {aircc,slurm}_archive/... runtime-comparison runs, old helper scripts

/mnt/data4t/models_for_experiments/
├── convnext_base/madry/l2/convnext_base_l2_4_init1.pth.tar    -> ../../../../models/...
├── convnext_base/baseline/convnext_base_baseline_init0.pth.tar
├── swin_b/madry/l1/swin_b_l1_2_init1.pth.tar
└── manifest.csv          model, kind, which rule chose it, target sha256
```

**The arch is always in the name.** The Slurm ViT/Swin dirs are named
`swin_b/l2_4_init1` with the arch only in the parent, and all 31 `swin_b` leaf
names are also `vit_b_cvst` leaf names — so anything keyed on the leaf silently
merges the two lanes. `naming.canonical_name` normalises that away.

## Running a pass

Every pass is dry-run by default, idempotent, resumable, and gets its own tmux
session. Logs and lock/stamp files go to `slurm_job_manager/logs/reorg/`
(local disk, gitignored — never the CIFS share, where `flock` is not dependable).

```bash
tmux new -s ms_conflicts 'model_store/scripts/ms_run.sh conflicts; read'
model_store/scripts/ms_run.sh            # lists every pass
```

| pass | step | effect |
|---|---|---|
| `dupes` | 1 | QNAP duplicate report → `01_qnap_duplicates.{md,csv}` |
| `conflicts` | 2 | merge-decision list → `02_merge_decisions.{md,csv}` — **the gate** |
| `build` / `build-apply` | 3 | hardlink `models/` (0 bytes) |
| `backfill` / `backfill-apply` | 4 | pull what the QNAP has and data4t lacks (**real copies**); `-- --roots qnap-slurm` to limit the source |
| `promotions` | 5 | recover the `/mnt/data` promotion decisions → `05_promotions.{md,csv}` |
| `zoo` / `zoo-apply` | 6 | build the symlink tree |
| `stage-data4t[-apply]` | 7 | `mv` archive leftovers → `pending_deletion/<date>/` |
| `stage-data[-apply]` | 8 | `mv` verified `/mnt/data` duplicates → `pending_deletion/<date>/` |
| `census` / `legacy` | — | inventory and the `_legacy/` listing |

Steps 4 and 6 also run every Monday from `model_store/scripts/ms_weekly_sync.sh`
(route 2 above), so models that finish during the week appear in both trees without
anyone doing anything. That script takes `--dry-run`.

**Step 3 is not in the weekly chain.** It hardlinks from `/mnt/data4t/{slurm,aircc}_archive`,
and those trees are frozen — route 1 writes straight to the QNAP now, so nothing refreshes
them. It stays a manual pass. The consequence is that models arriving after the cutover land
in `models/` as real copies (`nlink == 1`) rather than hardlinks: unavoidable, since there is
no local archive left to link against, and still exactly one copy on `/dev/sda1`. Note this
weakens rule 3 below for those models — `nlink == 1` no longer *only* means "discard".

## Three things worth knowing before changing any of this

**1. mtime lies; content decides.** 70 AIRCC `model_best.pth.tar` files carry the
mtime `2026-08-10 11:12:0x` — written within seconds of each other by a bulk
rewrite, not by training. Sampled pairs that a newest-wins rule would have
resolved in AIRCC's favour turned out byte-identical to the Slurm copy. So size +
mtime is only a pre-filter (`hashes.same_content`) and sha256 is the arbiter. The
Step 1 report bears this out: of 99 models present in both QNAP archives, 91 are
pure mirrors and only 8 diverge at all.

**2. Never write *through* a hardlink.** After Step 3 every file in `models/` is
the same inode as the archive copy. `rsync` writes a temp file and renames, which
breaks the link and leaves the archive intact — but `--inplace` / `--append` would
modify the archive's bytes, corrupting the master. Verified both ways; see the
comment on `backfill.RSYNC_BASE`. Do not add those flags.

**3. `nlink == 1` is the discard test.** Because Step 3 hardlinks every keeper,
`find <archive> -type f -links 1` afterwards *is* the set of files nothing in
`models/` references. Step 7 stages exactly that set, so the filesystem decides
what is safe to remove rather than a list this package has to be trusted to
compute. Step 7 additionally refuses if it finds an unlinked keeper the approved
decisions do not explain, because that means Step 3 skipped a model.

## Nothing here deletes

Removals are a same-filesystem `mv` into `pending_deletion/<date>/`, with a
`MANIFEST.csv` and a `README.md`; the user erases that directory by hand. So
staging frees **zero** bytes on its own, by design — `rsync --delete` would be an
irreversible erase. `model_store.stage --unstage <root>` puts it all back.

The one `--delete` in the package prunes stale *symlinks* from
`models_for_experiments` (`build_experiments.sync_into_place`). It cannot touch a
checkpoint.

## Decomposition precedence

`census._decompose`, most authoritative first:

1. **the job-manager CSVs** — `arch, protocol, threat_norm, threat_eps, init` as
   explicit columns, joined on `model_name`. Authoritative only when a DB row
   proves that CSV launched the run: `aircc/.../csv/convnext_small.csv` is a plan
   that was never run there, yet 40 of its 312 rows name-collide with real Slurm
   convnext_small dirs.
2. **the model's own config** — `config_reader`, two generations: flat `args.yaml`
   (older, `experiment_name` is null so the name comes from `output_dir`) and
   nested `hydra_config.yaml`. All 174 convnext_small dirs have one; none is
   missing.
3. **the folder name** — cross-check only. Where it disagrees with the config the
   model goes to `_legacy/unparsed/` and is reported, never guessed.

## Which checkpoint gets blessed

`build_experiments._resolve_kind`, and every entry records which rule fired:

1. `jobs.best_checkpoint` — **basename only**. The column holds three different
   cluster roots (`/home/ashtomer/projects/ares/results`,
   `/groups/golan_neurogroup/.../advmodels/results`,
   `/shared/cycle2_bgu_golan_prj/.../ares/results`), two of which no longer exist.
2. the kind recovered from `/mnt/data/robustness_models` by sha256
   (`promotions`) — this is how convnext_small gets a decision at all, since
   neither DB has a row for it and none of its dirs has an
   `autoattack_eps_norm_scores.json`.
3. `best_checkpoint_for_threat()` over the AutoAttack sweep CSVs
   (`aircc/aircc_job_manager/best_checkpoint.py`) — needs pandas, which is why
   `ms_run.sh` uses the `ares` conda env rather than base python.
4. `model_best.pth.tar`, labelled `fallback` so it is never mistaken for a score.

## Tests

```bash
/home/tomer_a/miniconda3/envs/ares/bin/python -m pytest model_store/tests/ -q
```
