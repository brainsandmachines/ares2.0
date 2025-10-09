#!/bin/bash
#SBATCH --job-name=convnext_small_training
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=2              # change to 1 if you want single-GPU
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=outs/general/Linf_new_rtx%A_%a.out   # Output file (%j will be replaced with job ID)

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

cd ~/projects/ares/robust_training
# ===== Run training =====
python -m torch.distributed.launch --nproc_per_node=${NPROC} \
   adversarial_training.py \
  --configs ./train_configs/convnext_small.yaml \
  --attack-eps 8.0  \


echo "=== Done ==="

