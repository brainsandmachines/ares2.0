#!/bin/bash
#SBATCH --job-name=autoattack_eval
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --array=9-77%10
#SBATCH --output=/home/ashtomer/projects/ares/outs/autoattack_eval/%A_%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

python data_analysis/autoattack_array_eval.py \
  --models-root /groups/golan_neurogroup/bml_group/tomerash/advmodels/results/models \
  --val-dir /groups/golan_neurogroup/bml_group/datasets/imagenet/val \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --batch-size 128 \
  --num-batches 8 \
  --num-workers 6 \
  --seed 0 \
  --device cuda \
  --force
