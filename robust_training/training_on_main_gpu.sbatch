#!/bin/bash
#SBATCH --job-name=convnext_small_training
#SBATCH --time=7-00:00:00
#SBATCH --partition=main                 # main queue
#SBATCH --gpus=1                         # request 1 GPU
#SBATCH --constraint=rtx_6000            # GPUs with >24GB VRAM
#SBATCH --mem=48G                        # memory (optional override)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=outs/advtrain/convnextsmall_%A_%a.out
#SBATCH --array=1-300%8


nvidia-smi
# ---- Env once ----
module load anaconda
source activate ares
conda deactivate
conda activate ares

# ---- Go to repo root (so relative paths work) ----
cd /home/ashtomer/projects/ares/robust_training

cleanup() {
  if [ -f /tmp/current_model.txt ]; then
    MODEL_ID=$(cat /tmp/current_model.txt)
    echo "[INFO] Cleaning up... resetting model $MODEL_ID to waiting."
    python - <<PY
from adv_scheduler import TaskScheduler
sch = TaskScheduler("adv_scheduler.db")
sch._execute_sqlite("UPDATE models SET status = 'waiting' WHERE model_id = ?", ("$MODEL_ID",))
PY
echo "[INFO] Model $MODEL_ID reset to waiting at $(date)" >> outs/cleanup_events.log
  fi
}
trap cleanup TERM INT EXIT


# ---- GPU mem → batch size ----
echo "[INFO] Checking GPU memory..."

# Query total and free memory (remove commas)
read GPU_TOTAL GPU_FREE <<< $(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ',' || echo "0 0")

echo "[INFO] Detected GPU total: ${GPU_TOTAL} MB, free: ${GPU_FREE} MB"

# Decide batch size based on total capacity
if [ "$GPU_TOTAL" -ge 95000 ]; then
  BATCH_SIZE=512
elif [ "$GPU_TOTAL" -ge 47000 ]; then
  BATCH_SIZE=256
elif [ "$GPU_TOTAL" -ge 23000 ]; then
  BATCH_SIZE=128
else
  echo "[ERROR] GPU total memory too small (< 24 GB)"
  exit 1
fi

# Require at least 90% free
REQUIRED_FREE=$(( GPU_TOTAL * 9 / 10 ))
if [ "$GPU_FREE" -lt "$REQUIRED_FREE" ]; then
  echo "[ERROR] Not enough free GPU memory!"
  echo "[INFO] Required: ${REQUIRED_FREE} MB (90% of ${GPU_TOTAL}), Available: ${GPU_FREE} MB"
  echo "[HINT] Another job might still be using the GPU. Try again later."
  exit 1
fi

echo "[INFO] Sufficient free GPU memory detected."
echo "[INFO] Using batch size = ${BATCH_SIZE}"


# ---- Pick next model from scheduler ----
echo "[INFO] Selecting model from scheduler..."

MODEL_INFO=$(timeout 120s python - <<'PY' | tail -n 1
import json, os
from adv_scheduler import TaskScheduler

sch = TaskScheduler("adv_scheduler.db")

model = sch.claim_next_waiting_model(cooldown_minutes=10)

if not model:
    print("NONE")
else:
    arch_name = "convnext_small"
    constraint_val = model["constraint_val"]
    eps_str = str(int(constraint_val)) if float(constraint_val).is_integer() else str(constraint_val)
    safe_name = f"{arch_name}_eps-{eps_str}_{model['norm']}_seed-{model['init_id']}"
    path = f"/home/ashtomer/projects/ares/robust_training/results/{arch_name}/{safe_name}"

    print(json.dumps({
        "model_id": model["model_id"],
        "norm": model["norm"],
        "constraint_val": eps_str,
        "adv_train": bool(model["adv_train"]),
        "init_id": model["init_id"],
        "safe_name": safe_name,
        "checkpoint_path": os.path.join(path, "model_best.pth.tar"),
        "remaining_epochs": max(0, 200 - int(model["current_epoch"]))
    }))
PY
)

echo "[INFO] finished selecting model from scheduler..."

if [ -z "$MODEL_INFO" ]; then
  echo "[ERROR] Scheduler query timed out or returned nothing. Exiting safely."
  exit 1
fi

if [ "$MODEL_INFO" = "NONE" ]; then
  echo "[INFO] No waiting models found. Exiting."
  exit 0
fi

# ---- Parse JSON → shell vars ----
MODEL_ID=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['model_id'])")
echo "$MODEL_ID" > /tmp/current_model.txt
SAFE_NAME=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['safe_name'])")
NORM=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['norm'])")
CONS=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['constraint_val'])")
ADV_BOOL=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print('true' if d['adv_train'] else 'false')")
INIT=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['init_id'])")
CHECKPOINT=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['checkpoint_path'])")
EPOCHS=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['remaining_epochs'])")

echo "[INFO] Selected model:"
echo "  ID:          $MODEL_ID"
echo "  Norm:        $NORM"
echo "  Constraint:  $CONS"
echo "  AdvTrain:    $ADV_BOOL"
echo "  InitID:      $INIT"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Remaining epochs: $EPOCHS"

export HYDRA_FULL_ERROR=1

# ---- Run Hydra training ----
if [ -f "$CHECKPOINT" ]; then
  echo "[INFO] Resuming training from checkpoint..."
  python hydra_advtrain.py \
    attacks.attack_norm="$NORM" \
    attacks.attack_eps="$CONS" \
    attacks.advtrain="$ADV_BOOL" \
    model.experiment_num="$INIT" \
    model.resume="$CHECKPOINT" \
    model.experiment_name="$SAFE_NAME" \
    model.model_id="'$MODEL_ID'" \
    training.batch_size="$BATCH_SIZE" \
    training.epochs="$EPOCHS" \
    hydra.run.dir="results/convnext_small/${SAFE_NAME}"
else
  echo "[INFO] Starting new training..."
  python hydra_advtrain.py \
    attacks.attack_norm="$NORM" \
    attacks.attack_eps="$CONS" \
    attacks.advtrain="$ADV_BOOL" \
    model.experiment_num="$INIT" \
    model.experiment_name="$SAFE_NAME" \
    model.model_id="'$MODEL_ID'" \
    training.batch_size="$BATCH_SIZE" \
    training.epochs="$EPOCHS" \
    hydra.run.dir="results/convnext_small/${SAFE_NAME}"
fi

rm -f /tmp/current_model.txt
echo "[INFO] Training job completed."