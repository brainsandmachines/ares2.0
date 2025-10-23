#!/bin/bash
#SBATCH --job-name=test_adv_train
#SBATCH --time=7-00:00:00
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=outs/test_adv_train_loop_%A.out

# ---- GPU info ----
nvidia-smi

# ---- Environment ----
module load anaconda
source activate ares
conda deactivate
conda activate ares

# ---- Go to repo root ----
cd /home/ashtomer/projects/ares/robust_training

# ---- GPU mem → batch size ----
echo "[INFO] Checking GPU memory..."
read GPU_TOTAL GPU_FREE <<< $(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ',' || echo "0 0")
echo "[INFO] GPU total: ${GPU_TOTAL} MB, free: ${GPU_FREE} MB"

if [ "$GPU_TOTAL" -ge 95000 ]; then
  BATCH_SIZE=512
elif [ "$GPU_TOTAL" -ge 47000 ]; then
  BATCH_SIZE=256
elif [ "$GPU_TOTAL" -ge 23000 ]; then
  BATCH_SIZE=128
else
  BATCH_SIZE=50
fi

REQUIRED_FREE=$(( GPU_TOTAL * 9 / 10 ))
if [ "$GPU_FREE" -lt "$REQUIRED_FREE" ]; then
  echo "[ERROR] Not enough free GPU memory!"
  echo "[INFO] Required: ${REQUIRED_FREE} MB (90% of ${GPU_TOTAL}), Available: ${GPU_FREE} MB"
  exit 1
fi

echo "[INFO] Using batch size = ${BATCH_SIZE}"

# ---- Training configuration ----
EPOCHS=300
TRAIN_PATH="/home/ashtomer/datasets/imagenet_sample/train"
VAL_PATH="/home/ashtomer/datasets/imagenet_sample/val"

echo "[INFO] Starting adversarial training on ImageNet sample dataset"
echo "  Train path: ${TRAIN_PATH}"
echo "  Val path:   ${VAL_PATH}"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
export HYDRA_FULL_ERROR=1
# ---- Run training ----
python hydra_advtrain.py \
  dataset.train_dir="$TRAIN_PATH" \
  dataset.eval_dir="$VAL_PATH" \
  training.batch_size="$BATCH_SIZE" \
  training.epochs="$EPOCHS" \
  attacks.advtrain=true \
  model.experiment_name="test_adv_train"

echo "[INFO] Adversarial training job completed successfully."
