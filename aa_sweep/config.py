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

# Small files always carried along when staging a model: the CSVs bring the existing eps_norm rows
# so they get reused, and the selection json pins the same 1024 evaluation images.
STAGE_ALWAYS_GLOBS: tuple[str, ...] = (
    "autoattack_sweep_results*.csv",
    "autoattack_sweep_selection.json",
    "autoattack_eps_norm_scores.json",
    "summary.csv",
    "hydra_config.yaml",
    "runtime_config.yaml",
)

# --- clusters --------------------------------------------------------------------------------
SLURM_SSH_HOST = os.environ.get("AA_SWEEP_SLURM_HOST", "slurm")
SLURM_REPO = os.environ.get("AA_SWEEP_SLURM_REPO", "/home/ashtomer/projects/ares")
SLURM_MODELS_ROOT = os.environ.get("AA_SWEEP_SLURM_MODELS_ROOT", f"{SLURM_REPO}/results/models")
SLURM_USER = os.environ.get("AA_SWEEP_SLURM_USER", "ashtomer")
SBATCH_SCRIPT = os.environ.get("AA_SWEEP_SBATCH", "sbatches/aa_sweep_completion.sbatch")

SLURM_MOUNT = Path(os.environ.get("AA_SWEEP_SLURM_MOUNT", HOME / "slurm_mount"))
AIRCC_MOUNT = Path(os.environ.get("AA_SWEEP_AIRCC_MOUNT", HOME / "aircc_mount"))

SJM_DB = Path(
    os.environ.get("AA_SWEEP_SJM_DB", SLURM_MOUNT / "projects/ares/slurm_job_manager/jobs.sqlite")
)
AIRCC_DB = Path(
    os.environ.get(
        "AA_SWEEP_AIRCC_DB",
        AIRCC_MOUNT / "ashtomer/ares/aircc/aircc_job_manager/aircc_jobs.sqlite",
    )
)
AIRCC_MODELS_ROOT = Path(
    os.environ.get("AA_SWEEP_AIRCC_MODELS_ROOT", AIRCC_MOUNT / "ashtomer/ares/results/models")
)

# Local Botero archive of the AIRCC results tree, refreshed by the 03:00 backup cron. Staging reads
# come from here when verified fresh (see mirror.py) -- local disk instead of ~3.4 MB/s sshfs.
# Moved to /mnt/data4t in Aug 2026: the archive now keeps every checkpoint (~1.9TB) so it can
# outlive the cluster-side deletion, which does not fit on /mnt/data.
BACKUP_MIRROR = Path(
    os.environ.get("AA_SWEEP_BACKUP_MIRROR", "/mnt/data4t/aircc_archive/models")
)
BACKUP_LOG = Path(os.environ.get("AA_SWEEP_BACKUP_LOG", BACKUP_MIRROR / "backup.log"))
USE_MIRROR = os.environ.get("AA_SWEEP_USE_MIRROR", "1") != "0"

JOB_NAME_PREFIX = "aaswp"
SSH_TIMEOUT_SECONDS = int(os.environ.get("AA_SWEEP_SSH_TIMEOUT", "60"))
RSYNC_TIMEOUT_SECONDS = int(os.environ.get("AA_SWEEP_RSYNC_TIMEOUT", "7200"))


def job_name(model_name: str, checkpoint_kind: str) -> str:
    """Slurm job name for one (model, kind) unit of work.

    Model names can be nested (``vit_b_cvst/linf_1_init1``); flatten so the name round-trips
    through ``squeue -o %j`` for the resubmission-dedupe check.
    """
    slug = model_name.strip("/").replace("/", "__")
    return f"{JOB_NAME_PREFIX}_{slug}_{checkpoint_kind}"


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
# Botero's own RTX 4090 is a GPU slot for this sweep alongside the BGU cluster. It runs strictly
# one job at a time (see botero_runner.py) against models that are already on local disk, and its
# results stay local -- they are never pushed back to either cluster, so census.kind_status() reads
# them as a third source.
BOTERO_DB = Path(os.environ.get("AA_SWEEP_BOTERO_DB", Path(__file__).resolve().parent / "botero_queue.sqlite"))
# Queue depth is a backlog, not a concurrency level -- the lane still runs exactly one job at a
# time. It has to cover the gap between nightly top-ups: at the measured ~7.4 h per (model, kind)
# the lane finishes ~3.2 units a day and the 21:30 cron refills only once, so 7 is a bit over two
# days of work -- enough to ride out a missed cron night without hoarding units the cluster could
# have run sooner.
BOTERO_SLOTS = int(os.environ.get("AA_SWEEP_BOTERO_SLOTS", "7"))

# Searched in order; the first directory holding both the checkpoint and the selection json wins.
# Local disk before the QNAP CIFS share. The two archives are NOT cleanly split by originating
# cluster (convnext_base_v1_l2_2_init1 lives under aircc_archive on the QNAP but slurm_archive on
# /mnt/data4t), so all four roots have to be searched.
BOTERO_MODEL_ROOTS: tuple[Path, ...] = tuple(
    Path(p) for p in os.environ.get(
        "AA_SWEEP_BOTERO_ROOTS",
        "/mnt/data4t/slurm_archive/models:/mnt/data4t/aircc_archive/models:"
        "/mnt/botero/slurm_archive/models:/mnt/botero/aircc_archive/models",
    ).split(":") if p
)
# Where the small artifacts are echoed after a local run. The QNAP mirror cron is currently
# disabled, so the runner copies the KB-sized CSV/PNG across itself.
BOTERO_QNAP_ROOT = Path(os.environ.get("AA_SWEEP_BOTERO_QNAP", "/mnt/botero"))
BOTERO_LOCAL_ROOT = Path(os.environ.get("AA_SWEEP_BOTERO_LOCAL", "/mnt/data4t"))

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
