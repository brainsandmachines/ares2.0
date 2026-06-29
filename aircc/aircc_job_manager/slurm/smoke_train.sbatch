#!/bin/bash
# AIRCC smoke test: launch ONE 1-epoch training in the v2 image via Pyxis to
# confirm the container + training entrypoint + dataset (+machine=aircc) all work,
# and that wandb logs to the adv_train_aircc project. Trains a tiny clean run
# (no adv, no final_eval) and asserts a checkpoint was written.
#
#   sbatch aircc/aircc_job_manager/slurm/smoke_train.sbatch
#
# NOTE: the v2 image ref contains a '#' (Nebius CR), which a `#SBATCH
# --container-image=` directive TRUNCATES at the '#'. So pass the IMAGE on the
# sbatch CLI (quoting preserves the '#'); MOUNTS stay a directive below. Submit:
#   sbatch --container-image='docker://cr.me-west1.nebius.cloud#i00xvqa4fe98d1s7pc/ares-train:v2' \
#          aircc/aircc_job_manager/slurm/smoke_train.sbatch
# The whole job runs in the sbatch-level container (no srun), so WORLD_SIZE is
# unset and dist.py takes the single-process path automatically.
#
#SBATCH --job-name=aircc_smoke
#SBATCH --partition=sandbox
#SBATCH --qos=sandbox_owner_780
#SBATCH --account=cycle2_bgu_golan_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --container-mounts=/shared/cycle2_bgu_golan_prj:/shared/cycle2_bgu_golan_prj
#SBATCH --output=/shared/cycle2_bgu_golan_prj/ashtomer/ares/aircc/aircc_job_manager/logs/smoke_%j.out
#SBATCH --error=/shared/cycle2_bgu_golan_prj/ashtomer/ares/aircc/aircc_job_manager/logs/smoke_%j.err

set -euo pipefail

REPO="${AIRCC_REPO:-/shared/cycle2_bgu_golan_prj/ashtomer/ares}"
AIRCC_IMAGE="${AIRCC_IMAGE:-docker://cr.me-west1.nebius.cloud#i00xvqa4fe98d1s7pc/ares-train:v2}"
AIRCC_MOUNTS="${AIRCC_MOUNTS:-/shared/cycle2_bgu_golan_prj:/shared/cycle2_bgu_golan_prj}"
mkdir -p "$REPO/aircc/aircc_job_manager/logs"

cd "$REPO"                               # container default workdir is /workspace
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export NAME="aircc_smoke_${SLURM_JOB_ID}"
export OUT="$REPO/results/aircc_smoke"
export WANDB_PROJECT=adv_train_aircc     # validate the wandb project end-to-end (online)
export AIRCC_MEM_FRACTION=0.47           # exercise the per-process memory cap
# wandb auth must be visible in the container: WANDB_API_KEY env (exported before
# sbatch) or a ~/.netrc mounted via AIRCC_MOUNTS.

echo "[smoke] image=$AIRCC_IMAGE"
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
