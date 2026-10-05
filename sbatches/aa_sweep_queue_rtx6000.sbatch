#!/bin/bash
# aa_sweep queue array for the rtx6000 partition. Same body and DB as the `main` array
# (aa_sweep/scripts/aa_queue_task.sh); the two lanes drain one queue.
#
# --qos=golan-neuro is required (the partition has DenyQos=normal). It is the same QOS the sjm
# training arrays run under, so this array shares the account's GPU cap with training.
#
# Batch size and conda env are picked by the task body from the card it lands on (48GB here, so
# 128 x 8 on tomer_advtrain) -- same 1024 images either way; an OOM at 128 falls back to 32 in-task.
#
#   sbatch --dependency=singleton sbatches/aa_sweep_queue_rtx6000.sbatch
#
# One-off array on the 96GB rtx_pro_6000 partition (same QOS; the body switches to
# tomer_advtrain_pro by card size). Not a feed-managed lane: nothing resubmits it.
#   sbatch --partition=rtx_pro_6000 --job-name=aaq-pro sbatches/aa_sweep_queue_rtx6000.sbatch
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
