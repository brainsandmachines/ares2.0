#!/bin/bash
# Slurm job-manager array for the rtx_pro_6000 partition (96 GB GPUs).
# One task = one GPU = one model: requeue dead owners -> claim one row -> run the
# single-process lifecycle (train -> final_eval AA -> plot) -> exit.
#
# Adjust GPU count by editing --array=1-200%K (K = concurrent GPUs) and resubmit.
#
#   sbatch slurm_job_manager/slurm/manager_rtx_pro_6000.sbatch
#
#SBATCH --job-name=sjm-manager
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gres=gpu:1
#SBATCH --array=1-200%6
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/sjm/%A/%a.out
set -uo pipefail

REPO="${SJM_CLUSTER_REPO:-/home/ashtomer/projects/ares}"
cd "$REPO" || exit 1
mkdir -p "$REPO/outs/sjm"

source "$REPO/sbatches/train_launcher_lib.sh"
module load anaconda
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
source activate "$(select_train_env "$GPU_TOTAL")"

export SJM_DB="${SJM_DB:-$REPO/slurm_job_manager/jobs.sqlite}"
export MODELS_ROOT="${MODELS_ROOT:-$REPO/results/models}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export SJM_TRAP_FILE="$(mktemp)"
# rtx_pro_6000 has 96 GB: use the CSV batch size as-is (divisor 1).
export SJM_BATCH_DIVISOR="${SJM_BATCH_DIVISOR:-1}"

# On Slurm time-limit, --signal delivers SIGTERM ~120s early to THIS shell:
# release the in-flight row so the next task resumes it, then exit (Slurm tears
# down the training process on script exit; last.pth.tar is the resume point).
release_and_exit() {
    python -m slurm_job_manager.release "$(cat "$SJM_TRAP_FILE" 2>/dev/null)" || true
    rm -f "$SJM_TRAP_FILE"
    exit 143
}
trap release_and_exit TERM

echo "[sjm] task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION gpu=${GPU_TOTAL}MB db=$SJM_DB"
python -m slurm_job_manager.controller &
CHILD=$!
wait "$CHILD"
status=$?
# After a clean exit, prune the finished model's rolling checkpoints (keep
# best/best_adv/last + top-3 by eval_top1). The controller wrote the claimed
# model name to SJM_TRAP_FILE; it is empty on a "no claimable work" exit.
if [ "$status" -eq 0 ]; then
    model_name="$(cat "$SJM_TRAP_FILE" 2>/dev/null || true)"
    if [ -n "$model_name" ] && [ -f "${MODELS_ROOT}/${model_name}/summary.csv" ]; then
        echo "[INFO] Running checkpoint pruning for $model_name"
        python "$REPO/results/models/prune_model_checkpoints.py" \
            "${MODELS_ROOT}/${model_name}" \
            --keep-n 3
        echo "[INFO] Checkpoint pruning finished with status $?"
    else
        echo "[INFO] No claimed model to prune (no-work exit)"
    fi
else
    echo "[INFO] Skipping checkpoint pruning because training failed or was killed"
fi
rm -f "$SJM_TRAP_FILE"
echo "[sjm] controller exit=$status"
exit "$status"
