#!/bin/bash
# AIRCC job-manager Slurm array on the B200 sandbox partition, run inside the v2
# NGC image via Pyxis. Up to 16 concurrent tasks = 16 B200 GPUs; each task runs
# ONE job_manager.py on ONE GPU and drives 2 training lifecycles. Effectively
# infinite; restartable at any time (the SQLite DB is the state).
#
#   1) generate CSVs once (committed):  python -m aircc.aircc_job_manager.generate_csvs
#   2) seed selectively (after review):  python -m aircc.aircc_job_manager.seed_db \
#                                           --db "$AIRCC_DB" --arch convnext_base --init 1
#   3) submit (image on the CLI -- its '#' can't live in a #SBATCH directive):
#        sbatch --container-image='docker://cr.me-west1.nebius.cloud#i00xvqa4fe98d1s7pc/ares-train:v2' \
#               aircc/aircc_job_manager/slurm/job_manager.sbatch
#
#SBATCH --job-name=aircc_jm
#SBATCH --array=1-200%16
#SBATCH --partition=sandbox
#SBATCH --qos=sandbox_owner_780
#SBATCH --account=cycle2_bgu_golan_prj
#SBATCH --gres=gpu:1
#SBATCH --container-mounts=/shared/cycle2_bgu_golan_prj:/shared/cycle2_bgu_golan_prj
#SBATCH --output=/shared/cycle2_bgu_golan_prj/ashtomer/ares/aircc/aircc_job_manager/logs/jm_%A_%a.out
#SBATCH --error=/shared/cycle2_bgu_golan_prj/ashtomer/ares/aircc/aircc_job_manager/logs/jm_%A_%a.err
# No #SBATCH --time: run until the partition max / cancellation.
# The whole job runs in the sbatch-level container (no srun), so WORLD_SIZE is
# unset; the 2 training procs per GPU run single-process (lifecycle also forces it).

set -euo pipefail

REPO="${AIRCC_REPO:-/shared/cycle2_bgu_golan_prj/ashtomer/ares}"
cd "$REPO"
mkdir -p aircc/aircc_job_manager/logs

# Shared, container-visible state.
export AIRCC_DB="${AIRCC_DB:-$REPO/aircc/aircc_job_manager/aircc_jobs.sqlite}"
export MODELS_ROOT="${MODELS_ROOT:-$REPO/results/models}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# wandb (project is set per-run to adv_train_aircc by the lifecycle):
# export WANDB_API_KEY=...   # or rely on ~/.netrc mounted into the container

echo "[aircc_jm] task ${SLURM_ARRAY_TASK_ID} on $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} db=$AIRCC_DB"
exec python -m aircc.aircc_job_manager.job_manager
