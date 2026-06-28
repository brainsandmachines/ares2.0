#!/bin/bash
# AIRCC smoke test: launch ONE 1-epoch training in the v2 NGC image via Pyxis to
# confirm the container + training entrypoint + dataset (+machine=aircc) all work
# before committing the full array. Trains a tiny clean run (no adv, no final_eval)
# and asserts a checkpoint was written.
#
#   sbatch aircc/aircc_job_manager/slurm/smoke_train.sbatch
#   # override the image:  sbatch --container-image=<.sqsh|docker://tag> smoke_train.sbatch
#
#SBATCH --job-name=aircc_smoke
#SBATCH --partition=sandbox
#SBATCH --qos=sandbox_owner_780
#SBATCH --account=cycle2_bgu_golan_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --container-image=docker://cr.me-west1.nebius.cloud#i00xvqa4fe98d1s7pc/ares-train:v2
#SBATCH --container-mounts=/shared/cycle2_bgu_golan_prj:/shared/cycle2_bgu_golan_prj
#SBATCH --output=/shared/cycle2_bgu_golan_prj/ashtomer/ares/aircc/aircc_job_manager/logs/smoke_%j.out
#SBATCH --error=/shared/cycle2_bgu_golan_prj/ashtomer/ares/aircc/aircc_job_manager/logs/smoke_%j.err

set -euo pipefail

REPO="${AIRCC_REPO:-/shared/cycle2_bgu_golan_prj/ashtomer/ares}"
cd "$REPO"
mkdir -p aircc/aircc_job_manager/logs

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MASTER_PORT=$((10000 + RANDOM % 50000))
export AIRCC_MEM_FRACTION=0.47           # exercise the per-process memory cap
export WANDB_MODE=offline                # don't pollute the project for a smoke test

NAME="aircc_smoke_${SLURM_JOB_ID}"
OUT="$REPO/results/aircc_smoke"
echo "[smoke] training one epoch -> $OUT/$NAME"

python -m robust_training.adversarial_training \
    +machine=aircc \
    model.experiment_name="$NAME" model.experiment_num=0 \
    attacks.advtrain=False \
    training.epochs=1 training.batch_size=128 \
    output_dir="$OUT" \
    final_eval=False

CKPT="$OUT/$NAME/last.pth.tar"
if [[ -f "$CKPT" ]]; then
    echo "[smoke] PASS: checkpoint written -> $CKPT"
else
    echo "[smoke] FAIL: no checkpoint at $CKPT" >&2
    exit 1
fi
