#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos golan-neuro
#SBATCH --time=14-00:00:00
#SBATCH --job-name=convnext_s_eps2_resume
#SBATCH --output=outs/general/convnext_s_eps2_resume-%j.out
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --mail-type=ARRAY_TASKS         # Notifications for job begin, end, and fail
#SBATCH --gpus=rtx_6000:1                      # Request 1 RTX 60000 GPU
#SBATCH --tasks=1

### Print some data to output file ###
echo "Job ID:                $SLURM_JOBID"
echo "Job name:	             $SLURM_JOB_NAME"
echo "Array Job ID:          $SLURM_ARRAY_JOB_ID"
echo "Task ID:               $SLURM_ARRAY_TASK_ID"
echo "Max Task Index:        $SLURM_ARRAY_TASK_MAX"
echo "Nodes allocated:       $SLURM_JOB_NODELIST"
echo "Running on node:       $SLURMD_NODENAME"
echo "Visible GPU IDs:       $CUDA_VISIBLE_DEVICES"

module load anaconda
source activate ares

# Comprehensive GPU and CUDA debugging
echo "=== System GPU Information ==="
nvidia-smi
echo ""

echo "=== CUDA Environment ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
which nvcc && nvcc --version || echo "nvcc not found"
echo "" 

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

# Detect how many GPUs were allocated (should be 1 here)
NPROC=$(python - <<'PY'
import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
echo "Using nproc_per_node=${NPROC}"

# ===== Resume training =====
CKPT=/home/ashtomer/projects/ares/robust_training/results/convnext_small_eps2_linf/last.pth.tar
OUTDIR=./results/convnext_small_eps2_linf   # keep writing to the same experiment folder

torchrun --nproc_per_node=${NPROC} \
  adversarial_training.py \
  --configs ./train_configs/convnext_small.yaml \
  --data-dir ~/datasets/imagenet \
  --output "${OUTDIR}" \
  --experiment convnext_small_eps2_linf \
  --resume "${CKPT}"

echo "=== Done ==="
