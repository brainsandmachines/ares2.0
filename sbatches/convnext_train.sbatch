#!/bin/bash
#SBATCH --job-name=convnext_small_training
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=2
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=outs/advtrain/convnextsmall_%A_%a.out

### --- GPU Sanity Check ---
echo "=== Checking GPU memory before starting ==="
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
THRESHOLD=500
if [ "$USED" -gt "$THRESHOLD" ]; then
  echo "GPU already using ${USED} MiB — requeuing job..."
  scontrol requeue $SLURM_JOBID
  exit 0
fi
echo "GPU looks clean (${USED} MiB used). Continuing..."

### --- Environment setup ---
module load anaconda
source activate ares

echo "[INFO] Checking GPU memory..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[INFO] Detected GPU memory: ${GPU_MEM} MB"

if [ "$GPU_MEM" -ge 47000 ]; then
  BATCH_SIZE=256
elif [ "$GPU_MEM" -ge 23000 ]; then
  BATCH_SIZE=128
else
  echo "[ERROR] GPU memory too small (< 24 GB)"
  exit 1
fi

echo "[INFO] Using batch size = ${BATCH_SIZE}"

# --- Python part: select next model from DB ---
echo "[INFO] Selecting model from scheduler..."

MODEL_INFO=$(python - <<'PY'
import sqlite3, json, os
from adv_scheduler import TaskScheduler

sch = TaskScheduler("adv_scheduler.db")
sch.requeue_stale_trainings(threshold_hours=10, max_epoch=200)

conn = sqlite3.connect("adv_scheduler.db")
c = conn.cursor()
c.execute("""
SELECT model_id, norm, constraint_val, adv_train, init_id, current_epoch
FROM models
WHERE status = 'waiting'
ORDER BY current_epoch DESC
LIMIT 1
""")
row = c.fetchone()
conn.close()

if not row:
    print("NONE")
else:
    model_id, norm, constraint_val, adv_train, init_id, epoch = row
    sch.update_status(model_id, "training")

    arch_name = "convnext_small"
    # Match your actual naming style: convnext_small_eps-2_linf_seed-1
    safe_name = f"{arch_name}_eps-{constraint_val}_{norm}_seed-{init_id}"
    path = f"/home/ashtomer/projects/ares/robust_training/results/{arch_name}/{safe_name}"
    remaining_epochs = max(0, 200 - int(epoch))  # compute 200 - current_epoch
    print(json.dumps({
        "model_id": model_id,
        "norm": norm,
        "constraint_val": constraint_val,
        "adv_train": adv_train,
        "init_id": init_id,
        "safe_name": safe_name,
        "checkpoint_path": os.path.join(path, "model_best.pth.tar"),
        "remaining_epochs": remaining_epochs
    }))
PY
)

if [ "$MODEL_INFO" == "NONE" ]; then
  echo "[INFO] No waiting models found. Exiting."
  exit 0
fi

# Parse model info from Python JSON
MODEL_ID=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['model_id'])")
SAFE_NAME=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['safe_name'])")
NORM=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['norm'])")
CONS=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['constraint_val'])")
ADV=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['adv_train'])")
INIT=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['init_id'])")
CHECKPOINT=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['checkpoint_path'])")
EPOCHS=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['remaining_epochs'])")

echo "[INFO] Selected model:"
echo "  ID:          $MODEL_ID"
echo "  Norm:        $NORM"
echo "  Constraint:  $CONS"
echo "  AdvTrain:    $ADV"
echo "  InitID:      $INIT"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Remaining epochs: $EPOCHS"


cd ~/projects/ares/robust_training
if [ -f "$CHECKPOINT" ]; then
  echo "[INFO] Resuming training from checkpoint..."
  module load anaconda
  source activate ares
  python hydra_advtrain.py \
    attacks.attack_norm="$NORM" \
    attacks.attack_eps="$CONS" \
    attacks.advtrain="$ADV" \
    model.experiment_num="$INIT" \
    model.resume="$CHECKPOINT" \
    training.batch_size="$BATCH_SIZE" \
    training.epochs="$EPOCHS" \
    hydra.run.dir="results/convnext_small/${SAFE_NAME}"
else
  echo "[INFO] Starting new training..."
  module load anaconda
  source activate ares
  python hydra_advtrain.py \
    attacks.attack_norm="$NORM" \
    attacks.attack_eps="$CONS" \
    attacks.advtrain="$ADV" \
    model.experiment_num="$INIT" \
    training.batch_size="$BATCH_SIZE" \
    training.epochs="$EPOCHS" \
    hydra.run.dir="results/convnext_small/${SAFE_NAME}"
fi

### ---Done---
