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
