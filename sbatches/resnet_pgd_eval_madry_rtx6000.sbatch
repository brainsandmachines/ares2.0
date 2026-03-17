#!/bin/bash
#SBATCH --job-name=PGD_eval_madry_rtx6000_strong
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/PGD_eval_madry_resnet50_rtx6000_strong/%A.out

set -euo pipefail

nvidia-smi

module load anaconda
source activate tomer_advtrain_pro

REPO_ROOT="/home/ashtomer/projects/ares"
cd "${REPO_ROOT}"

export VAL_DIR="${VAL_DIR:-/groups/golan_neurogroup/bml_group/tomerash/datasets/imagenet/val}"
export MODELS_ROOT="${MODELS_ROOT:-/groups/golan_neurogroup/bml_group/tomerash/madry_orig_robustmodels}"
export OUT_ROOT="${OUT_ROOT:-${MODELS_ROOT}/pgd_eval_resnet50_strong_l1_l2}"
export INPUT_MODE="${INPUT_MODE:-normalized}"
export PGD_EPS="${PGD_EPS:-1,3,5}"
export PGD_NORMS="${PGD_NORMS:-l1}"
export PGD_ATTACK_STEPS="${PGD_ATTACK_STEPS:-30}"
export PGD_ATTACK_RESTARTS="${PGD_ATTACK_RESTARTS:-7}"
export PGD_BATCH_SIZE="${PGD_BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

run_one_model () {
  local ckpt="$1"
  local model_stem
  model_stem="$(basename "${ckpt}" .ckpt)"
  local out_dir="${OUT_ROOT}/${model_stem}"
  mkdir -p "${out_dir}"

  echo "[INFO] running_model=${model_stem}"
  echo "[INFO] checkpoint=${ckpt}"
  echo "[INFO] out_dir=${out_dir}"
  echo "[INFO] pgd_norms=${PGD_NORMS} steps=${PGD_ATTACK_STEPS} restarts=${PGD_ATTACK_RESTARTS}"

  python resnet_pgd_exp/madry_resnet50_pgd_eval.py \
    --checkpoint "${ckpt}" \
    --val-dir "${VAL_DIR}" \
    --out-dir "${out_dir}" \
    --device cuda \
    --input-mode "${INPUT_MODE}" \
    --pgd-eps "${PGD_EPS}" \
    --pgd-norms "${PGD_NORMS}" \
    --pgd-attack-steps "${PGD_ATTACK_STEPS}" \
    --pgd-attack-restarts "${PGD_ATTACK_RESTARTS}" \
    --pgd-batch-size "${PGD_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"
}

run_one_model "${MODELS_ROOT}/resnet50_l2_eps0.ckpt"
run_one_model "${MODELS_ROOT}/resnet50_l2_eps0.01.ckpt"
run_one_model "${MODELS_ROOT}/resnet50_l2_eps0.03.ckpt"
run_one_model "${MODELS_ROOT}/resnet50_l2_eps0.05.ckpt"
