#!/bin/bash
#SBATCH --job-name=convnext_small_training
#SBATCH --time=7-00:00:00
#SBATCH --partition=main                 # main queue
#SBATCH --gpus=1                          # request 1 GPU
#SBATCH --constraint=rtx_6000            # GPUs with >24GB VRAM
#SBATCH --mem=48G                        # memory (optional override)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=outs/advtrain/convnextsmall/%A/%a.out
#SBATCH --array=1-100%8

nvidia-smi
# ---- Env once ----
module load anaconda
source activate ares
conda deactivate
conda activate ares

cd /home/ashtomer/projects/ares/robust_training
set -Eeuo pipefail
: "${SLURM_ARRAY_TASK_ID:=0}"

TMP_MODEL_FILE="/tmp/current_model_${SLURM_JOB_ID}.txt"

# ===================== CLEANUP BLOCK =====================
cleanup() {
  status=$?
  echo "[CLEANUP] Cleanup called with exit code $status"
  
  if [ -f "$TMP_MODEL_FILE" ]; then
    MODEL_ID=$(cat "$TMP_MODEL_FILE")
    echo "[INFO] Resetting model $MODEL_ID → waiting..."

    # 🔄 Use Model_scheduler instead of adv_scheduler
    python - <<PY
from model_scheduler import Model_scheduler
sch = Model_scheduler("model_scheduler.db")
sch._execute_sqlite(
    "UPDATE models SET status='waiting' WHERE model_id=?",
    ("$MODEL_ID",)
)
PY

    echo "[INFO] Model $MODEL_ID reset to waiting at $(date)" >> outs/cleanup_events.log
  fi
}

# Trap normal exits and signals
trap cleanup EXIT
trap cleanup INT TERM
# =========================================================

echo "[INFO] Checking GPU memory..."
GPU_INFO=$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr -d ',')
GPU_TOTAL=$(echo "$GPU_INFO" | awk '{print $1}' | sort -n | head -n 1)
GPU_FREE=$(echo "$GPU_INFO" | awk '{print $2}' | sort -n | head -n 1)
NUM_GPUS=$(echo "$GPU_INFO" | wc -l)

echo "[INFO] Detected ${NUM_GPUS} GPU(s)"
echo "[INFO] Smallest GPU total: ${GPU_TOTAL} MB, free: ${GPU_FREE} MB"

if [ "$GPU_TOTAL" -ge 95000 ]; then
  BATCH_PER_GPU=512
elif [ "$GPU_TOTAL" -ge 47000 ]; then
  BATCH_PER_GPU=256
elif [ "$GPU_TOTAL" -ge 23000 ]; then
  BATCH_PER_GPU=128
else
  echo "[ERROR] GPU total memory too small (< 24 GB)"
  exit 1
fi

REQUIRED_FREE=$(( GPU_TOTAL * 8 / 10 ))
if [ "$GPU_FREE" -lt "$REQUIRED_FREE" ]; then
  echo "[ERROR] Not enough free GPU memory!"
  echo "[INFO] Required: ${REQUIRED_FREE} MB (80%), Available: ${GPU_FREE} MB"
  exit 1
fi

TOTAL_BATCH=$(( BATCH_PER_GPU * NUM_GPUS ))
BATCH_SIZE=$BATCH_PER_GPU

echo "[INFO] GPU OK. Per-GPU batch: $BATCH_PER_GPU, Total batch: $TOTAL_BATCH"

# ===================== MODEL SELECTION =====================
echo "[INFO] Selecting model from scheduler..."

MODEL_INFO=$(timeout 120s python - <<'PY' | tail -n 1
import json, os
from model_scheduler import Model_scheduler

sch = Model_scheduler("model_scheduler.db")
model = sch.claim_next_waiting_model(cooldown_minutes=2)

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
        "epochs": model["epochs"],
        "grad_norm": model["grad_norm"],
    }))
PY
)
echo "[INFO] Finished selecting model."

if [ -z "$MODEL_INFO" ]; then
  echo "[ERROR] Scheduler query timed out or returned nothing."
  exit 1
fi

if [ "$MODEL_INFO" = "NONE" ]; then
  echo "[INFO] No waiting models found. Exiting."
  exit 0
fi

# ===================== JSON PARSING =====================
MODEL_ID=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['model_id'])")
SAFE_NAME=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['safe_name'])")
NORM=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['norm'])")
CONS=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['constraint_val'])")
ADV_BOOL=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print('true' if d['adv_train'] else 'false')")
INIT=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['init_id'])")
CHECKPOINT=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['checkpoint_path'])")
EPOCHS=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['epochs'])")
GRAD_NORM=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['grad_norm'])")

echo "$MODEL_ID" > "$TMP_MODEL_FILE"

echo "[INFO] Selected model:"
echo "  ID:          $MODEL_ID"
echo "  Norm:        $NORM"
echo "  Constraint:  $CONS"
echo "  AdvTrain:    $ADV_BOOL"
echo "  InitID:      $INIT"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Epochs left: $EPOCHS"
echo "  GradNorm:    $GRAD_NORM"
# =========================================================

export HYDRA_FULL_ERROR=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((10000 + SLURM_ARRAY_TASK_ID * 10))

# ===================== TRAINING RUN =====================
if [ -f "$CHECKPOINT" ]; then
  echo "[INFO] Resuming training..."
  torchrun --nproc_per_node=$NUM_GPUS --master-port=$MASTER_PORT hydra_advtrain.py \
    attacks.attack_norm="$NORM" \
    attacks.attack_eps="$CONS" \
    attacks.advtrain="$ADV_BOOL" \
    attacks.gradnorm="$GRAD_NORM" \
    model.experiment_num="$INIT" \
    model.resume="$CHECKPOINT" \
    model.experiment_name="$SAFE_NAME" \
    model.model_id="'$MODEL_ID'" \
    training.batch_size="$BATCH_SIZE" \
    training.epochs="$EPOCHS" \
    hydra.run.dir="results/convnext_small/${SAFE_NAME}"
else
  echo "[INFO] Starting new training..."
  torchrun --nproc_per_node=$NUM_GPUS --master-port=$MASTER_PORT hydra_advtrain.py \
    attacks.attack_norm="$NORM" \
    attacks.attack_eps="$CONS" \
    attacks.advtrain="$ADV_BOOL" \
    attacks.gradnorm="$GRAD_NORM" \
    model.experiment_num="$INIT" \
    model.experiment_name="$SAFE_NAME" \
    model.model_id="'$MODEL_ID'" \
    training.batch_size="$BATCH_SIZE" \
    training.epochs="$EPOCHS" \
    hydra.run.dir="results/convnext_small/${SAFE_NAME}"
fi

rm -f "$TMP_MODEL_FILE"
echo "[INFO] Training job completed."
