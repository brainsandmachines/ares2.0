#!/bin/bash

# Get current hour
CURRENT_HOUR=$(date +%H)

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate ares environment
conda activate ares

# Run the two Python scripts
python /home/ashtomer/projects/ares/robust_training/validate_db_vs_checkpoint.py
python /home/ashtomer/projects/ares/robust_training/requeue_daily.py

# Submit sbatch only at midnight (hour 0)
if [ "$CURRENT_HOUR" -eq 0 ]; then
    sbatch /home/ashtomer/projects/ares/robust_training/training_on_main_gpu.sbatch
fi
