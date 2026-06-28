#!/bin/bash
#SBATCH --job-name=orch-controller
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=1-200%6
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/orch_controller/%A_%a.out
#
# Controller array worker pool. One task = one GPU = one claimed model run.
# The botero monitor overrides --partition and --array (e.g. 1-200%8 on rtx6000);
# the directives above are defaults for a manual submit.
#
# Each task: requeue stale -> atomically claim one DB job -> run its remaining
# pipeline (train -> AA -> plot) -> exit. ORCH_DB is exported by the submitter
# (the monitor passes --export=ALL,ORCH_DB=<cluster path>); fall back to .env's
# cluster path for manual submits.

REPO="${ORCH_CLUSTER_REPO:-/home/ashtomer/projects/ares}"
cd "$REPO" || exit 1
mkdir -p "$REPO/outs/orch_controller"

nvidia-smi
source "$REPO/sbatches/train_launcher_lib.sh"
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
module load anaconda
source activate "$(select_train_env "$GPU_TOTAL")"

# ORCH_DB must point at the shared DB as seen from the cluster.
export ORCH_DB="${ORCH_DB:-$REPO/orchestrator/orchestrator.db}"
export ORCH_CLUSTER_REPO="$REPO"

echo "[controller.sbatch] task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION db=$ORCH_DB"
python -m orchestrator.controller --partition "$SLURM_JOB_PARTITION"
status=$?
echo "[controller.sbatch] controller exit=$status"
exit "$status"
