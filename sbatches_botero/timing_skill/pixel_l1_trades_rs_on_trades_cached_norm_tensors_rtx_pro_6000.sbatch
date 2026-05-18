#!/bin/bash
# Runtime-idea verification sbatch: emitted only after a local win for this protocol.
# This verifies 20 measured forward/backward training iterations on the target Slurm GPU.
# Batch size is 4x the local RTX 4090 batch size used for this protocol.
#SBATCH --job-name=pixel_l1_trades_rs_on_trades_cached_norm_tensors_rtx_pro_6000
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/timing_testing/pixel_l1_trades_rs_on_trades_cached_norm_tensors_rtx_pro_6000%a.out

set -euo pipefail

module load anaconda
source activate tomer_advtrain_pro

REPO_ROOT="/home/ashtomer/projects/ares"
cd "${REPO_ROOT}"

TRAIN_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/train/"
EVAL_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/val/"

mkdir -p "/home/ashtomer/projects/ares/outs/timing_testing/pixel_l1_trades_rs_on_trades_cached_norm_tensors_rtx_pro_6000"

bash "${REPO_ROOT}/.agents/skills/training-runtime-optimizer/scripts/benchmark_local.sh" \
  --protocol "pixel_l1_trades_rs_on" \
  --candidate "trades_cached_norm_tensors" \
  --output-dir "/home/ashtomer/projects/ares/outs/timing_testing/pixel_l1_trades_rs_on_trades_cached_norm_tensors_rtx_pro_6000" \
  --train-dir "${TRAIN_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --batch-sizes "448" \
  --warmup-iters "1" \
  --measured-iters "20" \
  --num-workers "6"
