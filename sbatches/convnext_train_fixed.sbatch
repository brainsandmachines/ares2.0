#!/bin/bash
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos golan-neuro                # Use the main partition
#SBATCH --time=14-00:00:00                         # 
#SBATCH --job-name=convnext_s_eps10_train            	# Job name
#SBATCH --output=outs/general/L1_new_rtx%A_%a.out     # Output file (%j will be replaced with job ID)
#SBATCH --mail-user=ashtomer@post.bgu.ac.il            # Email for notifications
#SBATCH --mail-type=ARRAY_TASKS         # Notifications for job begin, end, and fail
#SBATCH --gpus=rtx_pro_6000:2                      # Request 2 RTX 60000 GPUs
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

# How many GPUs did SLURM give us? (should match --gpus above)
NPROC=$(python - <<'PY'
import torch; print(torch.cuda.device_count())
PY
)
echo "Using nproc_per_node=${NPROC}"

# ===== Run training with torchrun =====
torchrun --nproc_per_node=${NPROC} \
   adversarial_training.py \
  --configs ./train_configs/convnext_small.yaml \
  --attack-eps 10.0  \


echo "=== Done ==="

