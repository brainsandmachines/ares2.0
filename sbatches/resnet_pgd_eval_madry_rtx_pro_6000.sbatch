#!/bin/bash
#SBATCH --job-name=PGD_eval_madry_rtxpro
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=7-9%3
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/PGD_eval_madry_resnet50_rtx_pro_6000/%A/%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain_pro

REPO_ROOT="/home/ashtomer/projects/ares"
cd "${REPO_ROOT}"

export VAL_DIR="${VAL_DIR:-/storage/test/bml_group/tomerash/datasets/imagenet/val}"
export MODELS_ROOT="${MODELS_ROOT:-/storage/test/bml_group/tomerash/madry_orig_robustmodels}"
export OUT_ROOT="${OUT_ROOT:-${MODELS_ROOT}/pgd_eval_resnet50}"
export INPUT_MODE="${INPUT_MODE:-normalized}"

bash resnet_pgd_exp/run_single_model_eval.sh "${SLURM_ARRAY_TASK_ID}"
