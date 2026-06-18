#!/bin/bash
# runtime_epoch_inspection
# -------------------------
# Array job: for every (protocol x convnext-arch) combo, run a capped-batch
# 1-epoch timing probe (steady-state batches, warmup excluded -> extrapolated to a
# full epoch) plus the end-of-epoch regular (clean) eval, and append one row to a
# shared CSV:
#   research/runtime-arch-eval/runtime_arch_eval.csv
#   columns: protocol,arch,bsz,optimizer,compile,fullepochruntime,train_runtime,eval_runtime
#
# 11 protocols x 3 archs = 33 tasks  ->  --array=0-32
#   p_idx = TASK_ID / 3   (protocol)
#   a_idx = TASK_ID % 3   (arch: small/base/large)
#
#SBATCH --job-name=runtime_epoch_inspection
#SBATCH --time=08:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/tomer_a/Documents/ares/outs/advtrain/runtime_arch_eval/%A/%a.out
#SBATCH --array=0-32

set -u

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}" || exit 1
echo "[INFO] REPO_ROOT=${REPO_ROOT}"

nvidia-smi
source "${REPO_ROOT}/sbatches/train_launcher_lib.sh"

# ---- GPU memory -> batch size / conda env ----
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[INFO] Detected GPU Memory: ${GPU_TOTAL} MB"
module load anaconda
source activate "$(select_train_env "$GPU_TOTAL")"

export WANDB_MODE=disabled
export MASTER_PORT=$((10000 + RANDOM % 50000))

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---- locate an ImageNet train/val pair ----
TRAIN_DIR=""
EVAL_DIR=""
candidate_pairs=(
    "/groups/golan_neurogroup/bml_group/datasets/imagenet/train|/groups/golan_neurogroup/bml_group/datasets/imagenet/val"
    "/storage/test/bml_group/tomerash/datasets/imagenet/train|/storage/test/bml_group/tomerash/datasets/imagenet/val"
    "${HOME}/datasets/imagenet/train|${HOME}/datasets/imagenet/val"
    "/mnt/data/datasets/imagenet/train|/mnt/data/datasets/imagenet/val"
)
for pair in "${candidate_pairs[@]}"; do
    train_candidate="${pair%%|*}"
    eval_candidate="${pair##*|}"
    if [[ -d "${train_candidate}" && -d "${eval_candidate}" ]]; then
        TRAIN_DIR="${train_candidate}"
        EVAL_DIR="${eval_candidate}"
        break
    fi
done
if [[ -z "${TRAIN_DIR}" || -z "${EVAL_DIR}" ]]; then
    echo "[ERROR] Could not find an ImageNet train/val directory pair."
    printf '  %s\n' "${candidate_pairs[@]}"
    exit 1
fi
echo "[INFO] TRAIN_DIR=${TRAIN_DIR}"
echo "[INFO] EVAL_DIR=${EVAL_DIR}"

# ---- protocol x arch matrix ----
# job-core fed to parse_train_job (eps fixed at 8; it does not affect runtime).
PROTOCOLS=(
    linf_8_init1
    l2_8_init1
    l1_8_init1
    linftrades_8_init1
    l2trades_8_init1
    l1trades_8_init1
    gradnorm_l1_8_init1
    gradnorm_l2_8_init1
    v1clean_l2_8_init1
    v1clean_l2trades_8_init1
    baseline_init1
)
# human-readable protocol label written to the CSV.
PLABELS=(
    madry_linf
    madry_l2
    madry_l1
    trades_linf
    trades_l2
    trades_l1
    gradnorm_l1
    gradnorm_l2
    v1madryl2
    v1tradesl2
    baseline
)
ARCHS=(small base large)
PREFIX=("" convnext_base_ convnext_large_)

n_arch=${#ARCHS[@]}
p_idx=$(( SLURM_ARRAY_TASK_ID / n_arch ))
a_idx=$(( SLURM_ARRAY_TASK_ID % n_arch ))

protocol_label="${PLABELS[$p_idx]}"
arch_label="${ARCHS[$a_idx]}"
jobname="${PREFIX[$a_idx]}${PROTOCOLS[$p_idx]}"

echo "[INFO] TASK_ID=${SLURM_ARRAY_TASK_ID} protocol=${protocol_label} arch=${arch_label} jobname=${jobname}"

# ---- derive training config from the job name (reuses production launcher logic) ----
parse_train_job "$jobname" "$GPU_TOTAL" || exit 1

echo "model_size      : $model_size"
echo "attack_norm     : $attack_norm"
echo "advtrain        : $advtrain"
echo "gradnorm        : $gradnorm"
echo "gradnorm_pen    : $gradnorm_penalty_norm"
echo "attack_crit     : $crit"
echo "attack_domain   : $attack_domain"
echo "is_v1           : $is_v1"
echo "batch_size      : $BATCH_SIZE"
echo "model_override  : $model_override"

RUN_DIR="${REPO_ROOT}/outs/advtrain/runtime_arch_eval/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
CSV_PATH="${REPO_ROOT}/research/runtime-arch-eval/runtime_arch_eval.csv"
mkdir -p "${RUN_DIR}"

cmd=(
    python -m robust_training.adversarial_training
    attacks.attack_norm="$attack_norm"
    attacks.advtrain="$advtrain"
    attacks.gradnorm="$gradnorm"
    attacks.gradnorm_penalty_norm="$gradnorm_penalty_norm"
    attacks.attack_criterion="$crit"
    attacks.attack_domain="$attack_domain"
    model.experiment_num="$experiment_num"
    model.resume=""
    training.batch_size="$BATCH_SIZE"
    training.epochs=1
    training.model_ema=false
    final_eval=false
    dataset.train_dir="$TRAIN_DIR"
    dataset.eval_dir="$EVAL_DIR"
    output_dir="$RUN_DIR"
    hydra.run.dir="${RUN_DIR}/hydra"
    runtime_probe=true
    runtime_probe_csv="$CSV_PATH"
    runtime_probe_protocol="$protocol_label"
    runtime_probe_arch="$arch_label"
)

# ---- production compile policy (mirrors sbatches/golan-trainmodels.sbatch) ----
if [[ "$advtrain" == "true" ]] && {
      [[ "$crit" == "madry" && "$attack_domain" == "pixel" ]] ||
      [[ "$crit" == "trades" && "$attack_domain" == "v1_feature" && "$GPU_TOTAL" -gt 90000 ]]
  }; then
      cmd+=("model.compile_model=True")
      echo "compile_model   : True"
else
      echo "compile_model   : False"
fi

if [[ -n "$model_override" ]]; then
    cmd+=("$model_override")
    cmd+=("model.v1_noise_mode=$v1_noise_mode")
fi

if [[ "$is_v1" == "true" && "$advtrain" == "true" ]]; then
    cmd+=("attacks.v1_attack_eps=$attack_eps")
elif [[ -n "$attack_eps" ]]; then
    cmd+=("attacks.attack_eps=$attack_eps")
fi

echo "[INFO] Running: ${cmd[*]}"
"${cmd[@]}"
status=$?
echo "[INFO] runtime probe exit status: $status"
exit "$status"
