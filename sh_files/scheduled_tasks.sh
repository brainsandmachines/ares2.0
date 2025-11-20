#!/bin/bash

# # -------  runs from login node 2 on the bgu cluster via cron every 6 hours -----
# # Get current hour
# CURRENT_HOUR=$(date +%H)
# printf "Current hour: %02d\n" "$CURRENT_HOUR"

# # Source the system Anaconda setup
# source /storage/modules/packages/anaconda/etc/profile.d/conda.sh

# # Activate your environment
# conda activate tomer_advtrain

# # Debug: show which Python is active
# echo "Using Python: $(which python)"
# echo "Python version: $(python --version)"

# # Run scripts
# python /home/ashtomer/projects/ares/job_manager/requeue_daily.py
# sbatch /home/ashtomer/projects/ares/sbatches/trainmodels_main.sbatch

# # Submit sbatch only at midnight
# if [ "$CURRENT_HOUR" -eq 0 ] || [ "$CURRENT_HOUR" -eq 12 ]; then
#     python /home/ashtomer/projects/ares/job_manager/validate_db_vs_checkpoint.py
# fi
