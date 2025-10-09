#!/usr/bin/env bash

#SBATCH --job-name=swin_base_training
#SBATCH --output=/home/ashtomer/logs/%x-%j.out
#SBATCH --time=6-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1              # change to 1 if you want single-GPU
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

# ===== Run training with torchrun =====
torchrun --nproc_per_node=${NPROC} \
   adversarial_training.py \
  --configs ./train_configs/swin_base_patch4_window7_224.yaml \


echo "=== Done ==="
