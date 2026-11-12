#!/bin/bash
# Slurm job-manager EVAL lane on the shared `main` partition. One job = one GPU = one
# trained model: requeue dead owners -> claim one `eval` row -> resume it at its final
# epoch (0 epochs train) so only final_eval runs (full AutoAttack sweep linf/l2/l1 x eps
# 1,2,4,6,8 on best/last/advbest, trained cell first -> DB best_checkpoint/best_score)
# plus shape-bias -> mark finished -> exit.
#
# Submitted automatically (python -m slurm_job_manager.eval_lane top_up): by the training
# lane right after a model's last epoch, by `reopen_eval --apply`, and by the hourly
# cron_release sweep (catch-up). By hand:
#
#   sbatch slurm_job_manager/slurm/manager_eval_main.sbatch
#
# --constraint/--exclude as in sbatches/aa_sweep_queue_main.sbatch: >=24GB cards only,
# never the *-pheno-* nodes (their second card is an 11GB gtx_1080 that --gpus=1 may hand
# out), and never dt-3090-01 / cs-3090-01 (no /groups mount: the ImageNet dirs are missing).
#
#SBATCH --job-name=sjm-eval
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint="rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000"
#SBATCH --exclude=cs-pheno-[01-12],ise-pheno-[01-12],dt-3090-01,cs-3090-01
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/sjm_eval/%j.out
set -uo pipefail

REPO="${SJM_CLUSTER_REPO:-/home/ashtomer/projects/ares}"
cd "$REPO" || exit 1
mkdir -p "$REPO/outs/sjm_eval"

VAL_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/val"
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
# Bail in seconds, before claiming, on a bad node.
if [[ -z "${GPU_TOTAL}" || "${GPU_TOTAL}" -lt 20000 ]]; then
    echo "[ERROR] GPU on $(hostname) has ${GPU_TOTAL:-unknown} MB < 20000 MB" >&2
    exit 1
fi
if [[ ! -d "$VAL_DIR" ]]; then
    echo "[ERROR] ImageNet val dir is not available on host $(hostname): $VAL_DIR" >&2
    exit 1
fi

source "$REPO/sbatches/train_launcher_lib.sh"
module load anaconda
source activate "$(select_train_env "$GPU_TOTAL")"

export SJM_DB="${SJM_DB:-$REPO/slurm_job_manager/jobs.sqlite}"
export MODELS_ROOT="${MODELS_ROOT:-$REPO/results/models}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export SJM_TRAP_FILE="$(mktemp)"
export SJM_LANE=eval
# AutoAttack geometry by card, always the same 1024 images (aa_sweep/scripts/aa_queue_task.sh):
# >=45GB cards keep the config default 128 x 8; 24GB cards OOM at 128, so 32 x 32.
if (( GPU_TOTAL >= 45000 )); then
    export SJM_EXTRA_OVERRIDES="${SJM_EXTRA_OVERRIDES:-}"
else
    export SJM_EXTRA_OVERRIDES="${SJM_EXTRA_OVERRIDES:-final_eval_aa_batch_size=32 final_eval_aa_max_batches=32}"
fi
# Nothing trains in this lane, but halve the CSV (96GB) batch anyway on smaller cards so
# the one warm-up batch the training script pulls never matters.
if (( GPU_TOTAL < 90000 )); then
    export SJM_BATCH_DIVISOR="${SJM_BATCH_DIVISOR:-2}"
fi

# On the time limit / preemption, --signal delivers SIGTERM ~120s early to THIS shell:
# release the row (a fully-trained row goes back to `eval`), then exit. The AA engine
# flushes every finished cell, so the next eval job resumes exactly there.
release_and_exit() {
    python -m slurm_job_manager.release "$(cat "$SJM_TRAP_FILE" 2>/dev/null)" || true
    rm -f "$SJM_TRAP_FILE"
    exit 143
}
trap release_and_exit TERM

echo "[sjm-eval] job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION host=$(hostname) gpu=${GPU_TOTAL}MB overrides='${SJM_EXTRA_OVERRIDES}' db=$SJM_DB"
python -m slurm_job_manager.controller --lane eval &
CHILD=$!
wait "$CHILD"
status=$?
rm -f "$SJM_TRAP_FILE"
echo "[sjm-eval] controller exit=$status"
exit "$status"
