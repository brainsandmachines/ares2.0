#!/bin/bash
#SBATCH --job-name=cleanup_db_test
#SBATCH --time=00:05:00
#SBATCH --partition=rtx6000
#SBATCH --gpus=1
#SBATCH --output=outs/cleanup_db_test_%A.out

cd /home/ashtomer/projects/ares/robust_training
module load anaconda
source activate ares

cleanup() {
  if [ -f /tmp/current_model.txt ]; then
    MODEL_ID=$(cat /tmp/current_model.txt)
    echo "[INFO] Cleaning up... resetting model $MODEL_ID to waiting."
    python - <<PY
from adv_scheduler import TaskScheduler
sch = TaskScheduler("adv_scheduler.db")
sch._execute_sqlite("UPDATE models SET status = 'waiting' WHERE model_id = ?", ("$MODEL_ID",))
PY
    echo "[INFO] Model $MODEL_ID reset at $(date)" >> outs/cleanup_events.log
  fi
}
trap cleanup TERM INT EXIT

echo "test_trap_001" > /tmp/current_model.txt

echo "[INFO] Simulating long job (sleep)..."
sleep 300
echo "[INFO] Done"
