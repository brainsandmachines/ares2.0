#!/bin/bash
#SBATCH --job-name=convnext_madry_1epoch_timing
#SBATCH --time=2-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/%j.convnext_madry_1epoch_timing.out

set -u

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/robust_training" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="/home/ashtomer/projects/ares"
fi
cd "${REPO_ROOT}" || exit 1

JOB_ID="${SLURM_JOB_ID:-local}"
TIMING_ROOT="${REPO_ROOT}/outs/advtrain/timing/${JOB_ID}"
RUN_ROOT="${TIMING_ROOT}/runs"
SUMMARY_TSV="${TIMING_ROOT}/timing_summary.tsv"
SETUP_LOG="${TIMING_ROOT}/setup.log"
mkdir -p "${TIMING_ROOT}" "${RUN_ROOT}"

{
    echo "[INFO] Repo root: ${REPO_ROOT}"
    echo "[INFO] Timing root: ${TIMING_ROOT}"
    date
    nvidia-smi
} >"${SETUP_LOG}" 2>&1

module load anaconda >>"${SETUP_LOG}" 2>&1
source activate tomer_advtrain_pro >>"${SETUP_LOG}" 2>&1

export WANDB_MODE=disabled
export MASTER_PORT=$((10000 + RANDOM % 50000))

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
    {
        echo "[ERROR] Could not find an ImageNet train/val directory pair."
        echo "[ERROR] Checked:"
        printf '  %s\n' "${candidate_pairs[@]}"
    } >"${SUMMARY_TSV}"
    cat "${SUMMARY_TSV}"
    exit 1
fi

printf "arch\tstatus\texit_code\tstart_time\tend_time\telapsed_seconds\telapsed_hms\tlog_path\n" >"${SUMMARY_TSV}"

format_hms() {
    local total_seconds="$1"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

run_arch() {
    local arch="$1"
    local batch_size="192"
    local log_path="${TIMING_ROOT}/${arch}.log"
    local start_epoch
    local end_epoch
    local elapsed
    local elapsed_hms
    local exit_code
    local status
    local start_time
    local end_time

    start_epoch="$(date +%s)"
    start_time="$(date -Is)"

    python -m robust_training.adversarial_training \
        "model=${arch}" \
        training.epochs=1 \
        "training.batch_size=${batch_size}" \
        attacks.advtrain=true \
        attacks.attack_criterion=madry \
        attacks.attack_norm=linf \
        attacks.attack_eps=1.0 \
        attacks.attack_it=3 \
        final_eval=false \
        "dataset.train_dir=${TRAIN_DIR}" \
        "dataset.eval_dir=${EVAL_DIR}" \
        "output_dir=${RUN_ROOT}/${arch}" \
        "model.experiment_name=${arch}_madry_1epoch_bsz${batch_size}" \
        "model.resume=" \
        >"${log_path}" 2>&1

    exit_code="$?"
    end_epoch="$(date +%s)"
    end_time="$(date -Is)"
    elapsed=$((end_epoch - start_epoch))
    elapsed_hms="$(format_hms "${elapsed}")"

    if [[ "${exit_code}" -eq 0 ]]; then
        status="ok"
    else
        status="failed"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${arch}" "${status}" "${exit_code}" "${start_time}" "${end_time}" \
        "${elapsed}" "${elapsed_hms}" "${log_path}" >>"${SUMMARY_TSV}"

    return "${exit_code}"
}

overall_status=0
for arch in convnext_large; do
    run_arch "${arch}" || overall_status=1
done

cat "${SUMMARY_TSV}"
echo
echo "setup_log=${SETUP_LOG}"

exit "${overall_status}"
