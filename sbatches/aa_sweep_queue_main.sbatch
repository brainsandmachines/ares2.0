#!/bin/bash
# aa_sweep queue array for the shared `main` partition. Each task claims ONE (model, kind) unit
# from aa_sweep/aa_queue.sqlite and runs its AutoAttack grid; the body is
# aa_sweep/scripts/aa_queue_task.sh (shared with the rtx6000 array).
#
# Normally submitted by the nightly feed (`python -m aa_sweep.submit` -> `cluster_queue feed
# --launch`), with --dependency=singleton so a new array waits for this lane's previous one.
# By hand:
#   sbatch --dependency=singleton sbatches/aa_sweep_queue_main.sbatch
#
#SBATCH --job-name=aaq-main
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint="rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000"
#SBATCH --exclude=cs-pheno-[01-12],ise-pheno-[01-12]
#SBATCH --array=1-200%8
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/aa_sweep/%A_%a.out

# --constraint/--exclude as in aa_sweep_completion.sbatch: >=24GB cards only, and never the
# *-pheno-* nodes whose second card is an 11GB gtx_1080 that --gpus=1 may hand out.

source /home/ashtomer/projects/ares/aa_sweep/scripts/aa_queue_task.sh
