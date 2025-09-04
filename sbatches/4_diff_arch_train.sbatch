#!/usr/bin/env bash
#SBATCH --job-name=resnet_vit_convnext_train
#SBATCH --output=/home/ashtomer/logs/%x-%j.out
#SBATCH --time=10-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=2                 # change to 1 if you want single-GPU
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il


module load anaconda
source activate ares

# === Repo & data roots ===
cd ~/projects/ares/robust_training

echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== Python & Torch check ==="
python - << 'PY'
import sys, torch
print("Python =", sys.version.split()[0])
print("Torch  =", torch.__version__)
print("CUDA?  =", torch.cuda.is_available())
if torch.cuda.is_available():
    n=torch.cuda.device_count()
    print("GPUs   =", n, "->", [torch.cuda.get_device_name(i) for i in range(n)])
PY

# How many GPUs did SLURM give us? (should match --gpus above)
NPROC=$(python - <<'PY'
import torch; print(torch.cuda.device_count())
PY
)
echo "Using nproc_per_node=${NPROC}"

# ===== Config list (runs sequentially) =====
CONFIGS=(
  "./train_configs/resnet50.yaml"
  "./train_configs/convnext_base.yaml"
  "./train_configs/vit_base_patch16_224.yaml"
  "./train_configs/swin_base_patch4_window7_224.yaml"
)

RUN_IDX=0
for CFG in "${CONFIGS[@]}"; do
  RUN_IDX=$((RUN_IDX+1))
  echo ""
  echo "================ RUN ${RUN_IDX}/4 :: ${CFG} ================"
  start_ts=$(date +"%Y-%m-%d %H:%M:%S")
  echo "Start time: ${start_ts}"

  torchrun --nproc_per_node="${NPROC}" \
    adversarial_training.py \
    --configs "${CFG}"

  end_ts=$(date +"%Y-%m-%d %H:%M:%S")
  echo "End time:   ${end_ts}"
  echo "================ COMPLETED RUN ${RUN_IDX}/4 ================"
done

echo "=== All runs finished ==="
