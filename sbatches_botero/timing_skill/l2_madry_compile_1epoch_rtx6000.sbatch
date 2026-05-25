#!/bin/bash
# Full-epoch L2 Madry compile verification on RTX 6000.
# Runs regular training for one epoch: init20 without compile, init30 with compile.
#SBATCH --job-name=l2_madry_compile_1epoch_rtx6000
#SBATCH --time=2-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/timing/full_epoch_compile/%j.rtx6000.out

set -u

REPO_ROOT="/home/ashtomer/projects/ares"
cd "${REPO_ROOT}" || exit 1

JOB_ID="${SLURM_JOB_ID:-local}"
GPU_KEY="rtx6000"
CONDA_ENV="tomer_advtrain"
BATCH_SIZE="256"
LRB="2.5e-4"
TIMING_ROOT="${REPO_ROOT}/outs/advtrain/timing/full_epoch_compile/${JOB_ID}_${GPU_KEY}"
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
source activate "${CONDA_ENV}" >>"${SETUP_LOG}" 2>&1

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
        printf '  %s\n' "${candidate_pairs[@]}"
    } | tee "${SUMMARY_TSV}"
    exit 1
fi

format_hms() {
    local total_seconds="$1"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

run_case() {
    local candidate="$1"
    local compile_model="$2"
    local experiment_num="$3"
    local run_dir="${RUN_ROOT}/${candidate}"
    local log_path="${TIMING_ROOT}/${candidate}.log"
    local start_epoch
    local end_epoch
    local elapsed
    local elapsed_hms
    local exit_code
    local status
    local start_time
    local end_time

    mkdir -p "${run_dir}"
    start_epoch="$(date +%s)"
    start_time="$(date -Is)"

    python -m robust_training.adversarial_training \
        attacks.attack_norm=l2 \
        attacks.advtrain=true \
        attacks.gradnorm=false \
        attacks.attack_criterion=madry \
        attacks.attack_domain=pixel \
        attacks.attack_eps=16 \
        model.experiment_num="${experiment_num}" \
        model.resume="" \
        model.compile_model="${compile_model}" \
        training.epochs=1 \
        training.batch_size="${BATCH_SIZE}" \
        lr_scheduler.lrb="${LRB}" \
        dataset.train_dir="${TRAIN_DIR}" \
        dataset.eval_dir="${EVAL_DIR}" \
        final_eval=false \
        output_dir="${run_dir}" \
        hydra.run.dir="${run_dir}/hydra" \
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

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${GPU_KEY}" "${candidate}" "${compile_model}" "${experiment_num}" "${status}" "${exit_code}" \
        "${start_time}" "${end_time}" "${elapsed}" "${elapsed_hms}" "${log_path}" >>"${SUMMARY_TSV}"

    return "${exit_code}"
}

printf "gpu\tcandidate\tcompile_model\texperiment_num\tstatus\texit_code\tstart_time\tend_time\telapsed_seconds\telapsed_hms\tlog_path\n" >"${SUMMARY_TSV}"

overall_status=0
run_case "no_compile_init20" "false" "20" || overall_status=1
run_case "compile_init30" "true" "30" || overall_status=1

cat "${SUMMARY_TSV}"
echo
echo "setup_log=${SETUP_LOG}"
exit "${overall_status}"
