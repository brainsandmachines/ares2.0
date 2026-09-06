"""Paths, hosts and the target grid for the AutoAttack sweep-completion driver.

Everything here is overridable by environment variable so the daily cron script and the tests can
point the driver at fixtures without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.path.expanduser("~"))

# --- the target grid -------------------------------------------------------------------------
# 3 norms x 5 eps on 3 checkpoint kinds = 45 cells per model. eps 12 was dropped from the goal.
NORMS: tuple[str, ...] = ("linf", "l2", "l1")
EPS_INPUTS: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0)
CHECKPOINT_KINDS: tuple[str, ...] = ("best", "last", "advbest")

# checkpoint filename and results CSV per kind -- must stay in step with
# data_analysis/autoattack_array_eval.py's CHECKPOINT_KIND_CANDIDATES / CHECKPOINT_KIND_SUFFIX.
CKPT_FILE_FOR_KIND: dict[str, str] = {
    "best": "model_best.pth.tar",
    "last": "last.pth.tar",
    "advbest": "model_best_adv.pth.tar",
}
CSV_FOR_KIND: dict[str, str] = {
    "best": "autoattack_sweep_results.csv",
    "last": "autoattack_sweep_results_last.csv",
    "advbest": "autoattack_sweep_results_advbest.csv",
}

SELECTION_JSON = "autoattack_sweep_selection.json"

# The two independent lanes. Which one a model belongs to is decided once, in plan.build_plan, and
# the split is disjoint: a model with a directory on the BGU cluster is the cluster's, everything
# else AIRCC finished is this machine's.
SLURM_LANE = "slurm"
BOTERO_LANE = "botero"

# --- clusters --------------------------------------------------------------------------------
SLURM_SSH_HOST = os.environ.get("AA_SWEEP_SLURM_HOST", "slurm")
SLURM_REPO = os.environ.get("AA_SWEEP_SLURM_REPO", "/home/ashtomer/projects/ares")
SLURM_MODELS_ROOT = os.environ.get("AA_SWEEP_SLURM_MODELS_ROOT", f"{SLURM_REPO}/results/models")
SLURM_USER = os.environ.get("AA_SWEEP_SLURM_USER", "ashtomer")
SBATCH_SCRIPT = os.environ.get("AA_SWEEP_SBATCH", "sbatches/aa_sweep_completion.sbatch")

SLURM_MOUNT = Path(os.environ.get("AA_SWEEP_SLURM_MOUNT", HOME / "slurm_mount"))

SJM_DB = Path(
    os.environ.get("AA_SWEEP_SJM_DB", SLURM_MOUNT / "projects/ares/slurm_job_manager/jobs.sqlite")
)

# --- AIRCC: a frozen job DB, and nothing else -------------------------------------------------
# The AIRCC allocation is finished and this machine no longer talks to it -- no `ssh aircc`, no
# `~/aircc_mount` sshfs. All that survives of it here is the *list of models it finished*, read
# from the frozen DB in the QNAP archive (127 finished of 323 rows, byte-for-byte the finished set
# the live DB last reported).
#
# That list is the Botero lane's candidate pool. The checkpoints themselves are NOT read from the
# archive: they are read from BOTERO_STORE_ROOT below, this machine's own local copy. Three of the
# 127 (convnext_base_{l2_cont4to6,linf_cont4to6,linf_cont4to8}_init0) never produced a checkpoint at
# all -- they sit under `models_failed/` -- so they simply never resolve to a local dir and are
# skipped.
AIRCC_ARCHIVE = Path(os.environ.get("AA_SWEEP_AIRCC_ARCHIVE", "/mnt/botero/aircc_archive"))
AIRCC_DB = Path(
    os.environ.get("AA_SWEEP_AIRCC_DB", AIRCC_ARCHIVE / "aircc_jobs_final_latest.sqlite")
)

JOB_NAME_PREFIX = "aaswp"
SSH_TIMEOUT_SECONDS = int(os.environ.get("AA_SWEEP_SSH_TIMEOUT", "60"))


def job_name(model_name: str, checkpoint_kind: str) -> str:
    """Slurm job name for one (model, kind) unit of work.

    Model names can be nested (``vit_b_cvst/linf_1_init1``); flatten so the name round-trips
    through ``squeue -o %j`` for the resubmission-dedupe check.
    """
    slug = model_name.strip("/").replace("/", "__")
    return f"{JOB_NAME_PREFIX}_{slug}_{checkpoint_kind}"


def own_job_names(model_name: str, checkpoint_kind: str) -> set[str]:
    """Every name this unit's job can appear under in ``squeue``.

    Two, not one, for a nested model. ``sbatches/aa_sweep_completion.sbatch`` renames the job to
    ``aaswp_$(basename AA_MODEL_DIR)_<kind>`` once it starts running, so
    ``vit_b_cvst/l2_cont4to6_init1`` is queued as ``aaswp_vit_b_cvst__l2_cont4to6_init1_best`` but
    *runs* as ``aaswp_l2_cont4to6_init1_best``. Matching only the queued form would let the nightly
    run submit a second job onto a CSV a running one already owns.
    """
    return {
        job_name(model_name, checkpoint_kind),
        job_name(model_name.rsplit("/", 1)[-1], checkpoint_kind),
    }


def parse_job_name(name: str) -> tuple[str, str] | None:
    """Inverse of :func:`job_name`: ``aaswp_<slug>_<kind>`` -> ``(model_name, kind)``.

    Returns None for anything that is not one of ours, so a `squeue -o %j` listing can be filtered
    and decoded in one pass. `__` round-trips back to the `/` of a nested sjm name.
    """
    if not name.startswith(f"{JOB_NAME_PREFIX}_"):
        return None
    body = name[len(JOB_NAME_PREFIX) + 1:]
    kind = body.rsplit("_", 1)[-1]
    if kind not in CHECKPOINT_KINDS or len(body) <= len(kind) + 1:
        return None
    return body[: -(len(kind) + 1)].replace("__", "/"), kind


# --- the Botero lane -------------------------------------------------------------------------
# Botero's own RTX 4090 is an independent sweep lane, not a helper for the cluster. It runs strictly
# one job at a time (see botero_runner.py) against this machine's own local models, and writes its
# results back into the same local dir. Nothing is copied to or from a cluster in either direction:
# whatever propagation is wanted is the weekly rsync's business, not this package's.
BOTERO_DB = Path(os.environ.get("AA_SWEEP_BOTERO_DB", Path(__file__).resolve().parent / "botero_queue.sqlite"))
# Queue depth is a backlog, not a concurrency level -- the lane still runs exactly one job at a
# time. It has to cover the gap between nightly top-ups: at the measured ~7.4 h per (model, kind)
# the lane finishes ~3.2 units a day and the 21:30 cron refills only once, so 7 is a bit over two
# days of work -- enough to ride out a missed cron night without hoarding units the cluster could
# have run sooner.
BOTERO_SLOTS = int(os.environ.get("AA_SWEEP_BOTERO_SLOTS", "7"))

# This machine's own models: the model_store curated tree, laid out <arch>/<name>/ and holding
# every keeper checkpoint plus the AutoAttack CSVs, selection json and plots. Local SATA, not CIFS.
#
# Model names from the AIRCC DB are flat (`convnext_base_baseline_init0`) while the tree is nested
# one level under an architecture dir, so resolution indexes the tree by directory basename rather
# than joining the name onto the root -- see botero.resolve_model_dir. Verified safe: 331 model dirs,
# 331 distinct basenames, no collisions.
BOTERO_STORE_ROOT = Path(os.environ.get("AA_SWEEP_BOTERO_STORE", "/mnt/data4t/models"))
# Bookkeeping dirs of the store, not models.
BOTERO_STORE_SKIP: frozenset[str] = frozenset({"_legacy", "_meta"})

BOTERO_VAL_DIR = Path(os.environ.get("AA_SWEEP_BOTERO_VAL_DIR", "/mnt/data/datasets/imagenet/val"))
BOTERO_PYTHON = os.environ.get("AA_SWEEP_BOTERO_PYTHON", str(HOME / "miniconda3/envs/ares/bin/python"))
BOTERO_REPO = Path(os.environ.get("AA_SWEEP_BOTERO_REPO", HOME / "Documents/ares"))
BOTERO_LOG_DIR = Path(os.environ.get("AA_SWEEP_BOTERO_LOG_DIR", Path(__file__).resolve().parent / "logs/botero"))

# 1024 images, always -- the invariant the whole sweep rests on. `batch_size * num_batches` must
# equal this so a hand-set batch size can never quietly change how many images are attacked.
# (With a selection json present the count is pinned by the json regardless, and the loader is
# shuffle=False, so batching only regroups the *same* images in the *same* order. num_batches is
# then only metadata -- but it is recorded in the CSV, so it still has to be right.)
BOTERO_TOTAL_IMAGES = 1024

# 32, same as sbatches/aa_sweep_completion.sbatch. Raising it was measured on this 4090 and
# rejected on both counts (2026-08-31, full standard AutoAttack on one batch at eps=1, the memory
# worst case since the most points survive into fab-t/square):
#
#   bsz 128  vit_b_cvst        OK, peak 17.9 GiB
#   bsz 128  swin_b            OOM at 23.0 GiB of 23.5
#   bsz 128  convnext_base_v1  OOM at 23.0 GiB
#   bsz 128  convnext_base     OOM at 23.0 GiB
#
# Three of the four architectures in this campaign do not fit, and the ViT that does is no faster:
# ~7.8 s/image at 128 against ~7.1 s/image for the comparable bsz-32 cell. AutoAttack's cost is
# APGD iterations and restarts *per image*, which batching does not change, and the card is already
# compute-bound at 32. So a bigger batch buys nothing here -- it only removes the headroom that lets
# you use the GPU for anything else while a sweep runs.
BOTERO_BATCH_SIZE = int(os.environ.get("AA_SWEEP_BOTERO_BATCH_SIZE", "32"))
BOTERO_NUM_BATCHES = int(os.environ.get("AA_SWEEP_BOTERO_NUM_BATCHES", "32"))
BOTERO_NUM_WORKERS = int(os.environ.get("AA_SWEEP_BOTERO_NUM_WORKERS", "6"))
BOTERO_MAX_ATTEMPTS = int(os.environ.get("AA_SWEEP_BOTERO_MAX_ATTEMPTS", "3"))
