#!/bin/bash
# Get current hour
CURRENT_HOUR=$(date +%H)

# Source the system Anaconda setup
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh

# Activate your environment
conda activate ares

# Debug: show which Python is active
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Run scripts
python /home/ashtomer/projects/ares/robust_training/validate_db_vs_checkpoint.py
python /home/ashtomer/projects/ares/robust_training/requeue_daily.py

# Submit sbatch only at midnight
if [ "$CURRENT_HOUR" -eq 0 ]; then
    sbatch /home/ashtomer/projects/ares/robust_training/training_on_main_gpu.sbatch
fi
