#!/bin/bash
#SBATCH --job-name=convnext_small_training
#SBATCH --time=7-00:00:00
#SBATCH --partition=main                 # main queue
#SBATCH --gpus=1                         # request 1 GPU
#SBATCH --constraint=rtx_4090            # GPUs with >24GB VRAM
#SBATCH --mem=48G                        # memory (optional override)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=outs/advtrain/convnextsmall_%A_%a.out
#SBATCH --error=outs/advtrain/convnextsmall_%A_%a.err
#SBATCH --array=1-10%8


# ---- Env once ----
source ~/.bashrc
module load anaconda
conda activate ares

# ---- Go to repo root (so relative paths work) ----
cd /home/ashtomer/projects/ares/robust_training

# ---- GPU mem → batch size ----
echo "[INFO] Checking GPU memory..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 || echo 0)
echo "[INFO] Detected GPU memory: ${GPU_MEM} MB"

if [ "$GPU_MEM" -ge 95000 ]; then
  BATCH_SIZE=512
elif [ "$GPU_MEM" -ge 47000 ]; then
  BATCH_SIZE=256
elif [ "$GPU_MEM" -ge 23000 ]; then
  BATCH_SIZE=128
else
  echo "[ERROR] GPU memory too small (< 24 GB)"
  exit 1
fi
echo "[INFO] Using batch size = ${BATCH_SIZE}"

# ---- Pick next model from scheduler ----
echo "[INFO] Selecting model from scheduler..."

MODEL_INFO=$(python - <<'PY' | tail -n 1
import json, os
from adv_scheduler import TaskScheduler

sch = TaskScheduler("adv_scheduler.db")
sch.requeue_stale_trainings(threshold_hours=10, max_epoch=200)

model = sch.claim_next_waiting_model(max_epoch=200)

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

if [ "$MODEL_INFO" = "NONE" ]; then
  echo "[INFO] No waiting models found. Exiting."
  exit 0
fi

# ---- Parse JSON → shell vars ----
MODEL_ID=$(echo "$MODEL_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['model_id'])")
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
