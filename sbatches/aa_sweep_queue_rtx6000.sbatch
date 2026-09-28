#!/bin/bash
# aa_sweep queue array for the rtx6000 partition. Same body and DB as the `main` array
# (aa_sweep/scripts/aa_queue_task.sh); the two lanes drain one queue.
#
# --qos=golan-neuro is required (the partition has DenyQos=normal). It is the same QOS the sjm
# training arrays run under, so this array shares the account's GPU cap with training.
#
#   sbatch --dependency=singleton sbatches/aa_sweep_queue_rtx6000.sbatch
#
#SBATCH --job-name=aaq-rtx6000
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gres=gpu:1
#SBATCH --array=1-200%8
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/aa_sweep/%A_%a.out

source /home/ashtomer/projects/ares/aa_sweep/scripts/aa_queue_task.sh
