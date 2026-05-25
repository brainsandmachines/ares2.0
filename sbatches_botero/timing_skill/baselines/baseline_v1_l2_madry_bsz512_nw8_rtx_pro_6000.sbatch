#!/bin/bash
# Slurm baseline timing sbatch for candidate comparison.
# This runs the unchanged benchmark setup: same protocol, batch size, workers,
# warmup, measured iterations, dataset roots, GPU partition, and conda env.
#SBATCH --job-name=baseline_v1_l2_madry_bsz512_nw8_rtx_pro_6000
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/timing_testing/slurm_baselines/baseline_v1_l2_madry_bsz512_nw8_rtx_pro_6000-%j.out

set -euo pipefail

module load anaconda
source activate tomer_advtrain_pro

REPO_ROOT="/home/ashtomer/projects/ares"
cd "${REPO_ROOT}"

TRAIN_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/train/"
EVAL_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/val/"

mkdir -p "/home/ashtomer/projects/ares/outs/timing_testing/slurm_baselines/rtx_pro_6000/v1_l2_madry/bsz_512_nw_8"

bash "${REPO_ROOT}/.agents/skills/training-runtime-optimizer/scripts/benchmark_local.sh" \
  --protocol "v1_l2_madry" \
  --candidate "baseline" \
  --output-dir "/home/ashtomer/projects/ares/outs/timing_testing/slurm_baselines/rtx_pro_6000/v1_l2_madry/bsz_512_nw_8" \
  --train-dir "${TRAIN_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --batch-sizes "512" \
  --warmup-iters "1" \
  --measured-iters "20" \
  --num-workers "8"
