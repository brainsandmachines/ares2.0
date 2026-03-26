#!/bin/bash
#SBATCH --job-name=PGD_eval_madry_resnet50
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=0-31%4
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/PGD_eval_madry_resnet50/%A/%a.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_train

REPO_ROOT="/home/ashtomer/projects/ares"
cd "${REPO_ROOT}"

# Override these at submit time if needed, e.g.:
# VAL_DIR=/some/imagenet/val MODELS_ROOT=/some/models sbatch sbatches/resnet_pgd_eval_madry.sbatch
export VAL_DIR="${VAL_DIR:-/groups/golan_neurogroup/bml_group/datasets/imagenet/val}"
export MODELS_ROOT="${MODELS_ROOT:-/groups/golan_neurogroup/bml_group/tomerash/madry_orig_robustmodels}"
export OUT_ROOT="${OUT_ROOT:-${MODELS_ROOT}/pgd_eval_resnet50}"
export INPUT_MODE="${INPUT_MODE:-auto}"

bash resnet_pgd_exp/run_single_model_eval.sh "${SLURM_ARRAY_TASK_ID}"
