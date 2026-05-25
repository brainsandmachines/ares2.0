#!/bin/bash
# Slurm baseline timing sbatch for candidate comparison.
# This runs the unchanged benchmark setup: same protocol, batch size, workers,
# warmup, measured iterations, dataset roots, GPU partition, and conda env.
#SBATCH --job-name=baseline_pixel_l2_madry_bsz256_nw6_rtx6000
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/timing_testing/slurm_baselines/baseline_pixel_l2_madry_bsz256_nw6_rtx6000-%j.out

set -euo pipefail

module load anaconda
source activate tomer_advtrain

REPO_ROOT="/home/ashtomer/projects/ares"
cd "${REPO_ROOT}"

TRAIN_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/train/"
EVAL_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/val/"

mkdir -p "/home/ashtomer/projects/ares/outs/timing_testing/slurm_baselines/rtx6000/pixel_l2_madry/bsz_256_nw_6"

bash "${REPO_ROOT}/.agents/skills/training-runtime-optimizer/scripts/benchmark_local.sh" \
  --protocol "pixel_l2_madry" \
  --candidate "baseline" \
  --output-dir "/home/ashtomer/projects/ares/outs/timing_testing/slurm_baselines/rtx6000/pixel_l2_madry/bsz_256_nw_6" \
  --train-dir "${TRAIN_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --batch-sizes "256" \
  --warmup-iters "1" \
  --measured-iters "20" \
  --num-workers "6"
