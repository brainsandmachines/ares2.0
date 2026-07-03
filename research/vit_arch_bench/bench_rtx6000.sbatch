#!/bin/bash
# vit_arch_bench (rtx6000, 48GB)
# -----------------------------------
# Array job: one task per new ViT-style arch (6 total). Each task:
#   1. finds the largest per-GPU batch size (multiples of 64) that survives a
#      real 1-batch training smoke under TRADES (the most memory-hungry of the
#      three protocols, two forwards + KL), so the chosen bsz is safe for all;
#   2. for each protocol (baseline / madry / trades) x compile (off / on), runs
#      a steady-state timing probe: 5 warmup + 20 measured batches, extrapolated
#      to a full-epoch estimate by the existing runtime_probe machinery;
#   3. appends one CSV row per (protocol, compile) combo to
#      research/vit_arch_bench/results_rtx6000.csv
#      (columns: protocol,arch,bsz,optimizer,compile,fullepochruntime,train_runtime,eval_runtime)
#
# Mirrors sbatches_botero/runtime_epoch_inspection.sbatch, but drives the new
# arches via Hydra model groups directly (no convnext job-name parsing) and
# times BOTH compile settings instead of the production compile policy.
#
#SBATCH --job-name=vit_arch_bench
#SBATCH --time=08:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/vit_arch_bench/%A/%a.out
#SBATCH --array=0-5

set -u

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}" || exit 1
echo "[INFO] REPO_ROOT=${REPO_ROOT}"

nvidia-smi
source "${REPO_ROOT}/sbatches/train_launcher_lib.sh"

is_oom_log() {
    local log_path="$1"
    grep -Eiq 'out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA out of memory|DefaultCPUAllocator: can.t allocate memory' "$log_path"
}

GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[INFO] Detected GPU Memory: ${GPU_TOTAL} MB"
module load anaconda
source activate "$(select_train_env "$GPU_TOTAL")"

export WANDB_MODE=disabled
export MASTER_PORT=$((10000 + RANDOM % 50000))

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---- ImageNet train/val ----
TRAIN_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/train"
EVAL_DIR="/groups/golan_neurogroup/bml_group/datasets/imagenet/val"
if [[ ! -d "${TRAIN_DIR}" || ! -d "${EVAL_DIR}" ]]; then
    echo "[ERROR] ImageNet train/val not found at ${TRAIN_DIR} / ${EVAL_DIR}"
    exit 1
fi
echo "[INFO] TRAIN_DIR=${TRAIN_DIR}"
echo "[INFO] EVAL_DIR=${EVAL_DIR}"

# ---- arch (array index) and protocols ----
# Hydra model-group names (robust_training/configs/model/<name>.yaml).
MODELS=(vit_s vit_m vit_s_cvst vit_m_cvst swin_t swin_b)
PROTOCOLS=(baseline madry trades)

arch="${MODELS[$SLURM_ARRAY_TASK_ID]}"
PARTITION="${SLURM_JOB_PARTITION:-rtx6000}"
echo "[INFO] TASK_ID=${SLURM_ARRAY_TASK_ID} arch=${arch} partition=${PARTITION}"

RUN_DIR="${REPO_ROOT}/outs/vit_arch_bench/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
CSV_PATH="${REPO_ROOT}/research/vit_arch_bench/results_${PARTITION}.csv"
mkdir -p "${RUN_DIR}"

# eps fixed at 8 (/255, normalized inside main); it does not affect runtime.
ATTACK_EPS=8

build_runtime_cmd() {
    local protocol="$1"     # baseline | madry | trades
    local compile_flag="$2" # True | False
    local batch_size="$3"
    local output_dir="$4"
    local csv_path="$5"
    local warmup_batches="$6"
    local measure_batches="$7"

    cmd=(
        python -m robust_training.adversarial_training
        model="$arch"
        model.experiment_num=1
        model.resume=""
        model.compile_model="$compile_flag"
        training.batch_size="$batch_size"
        training.epochs=1
        training.model_ema=false
        final_eval=false
        dataset.train_dir="$TRAIN_DIR"
        dataset.eval_dir="$EVAL_DIR"
        output_dir="$output_dir"
        hydra.run.dir="${output_dir}/hydra"
        runtime_probe=true
        runtime_probe_csv="$csv_path"
        runtime_probe_protocol="${protocol}"
        runtime_probe_arch="$arch"
        runtime_probe_warmup_batches="$warmup_batches"
        runtime_probe_measure_batches="$measure_batches"
    )

    if [[ "$protocol" == "baseline" ]]; then
        cmd+=(attacks.advtrain=false)
    else
        cmd+=(
            attacks.advtrain=true
            attacks.attack_criterion="$protocol"
            attacks.attack_norm=linf
            attacks.attack_eps="$ATTACK_EPS"
        )
    fi
}

# ---- 1. max-bsz search (multiples of 64) under TRADES, compile off ----
SMOKE_ROOT="${RUN_DIR}/batch_size_smoke"
SMOKE_SUMMARY="${SMOKE_ROOT}/batch_size_search.tsv"
mkdir -p "${SMOKE_ROOT}"
printf "batch_size\tstatus\texit_code\tlog_path\n" >"${SMOKE_SUMMARY}"

selected_batch_size=0
for candidate_bsz in $(seq 64 64 2048); do
    smoke_dir="${SMOKE_ROOT}/bsz_${candidate_bsz}"
    smoke_log="${smoke_dir}/smoke.log"
    mkdir -p "${smoke_dir}"

    build_runtime_cmd trades False "$candidate_bsz" "$smoke_dir" "${smoke_dir}/smoke_runtime.csv" 0 1
    echo "[INFO] Batch-size smoke (trades): bsz=${candidate_bsz}"
    "${cmd[@]}" >"${smoke_log}" 2>&1
    smoke_status=$?

    if [[ "$smoke_status" -eq 0 ]]; then
        selected_batch_size="$candidate_bsz"
        printf "%s\tok\t%s\t%s\n" "$candidate_bsz" "$smoke_status" "$smoke_log" >>"${SMOKE_SUMMARY}"
        continue
    fi

    if is_oom_log "$smoke_log"; then
        printf "%s\toom\t%s\t%s\n" "$candidate_bsz" "$smoke_status" "$smoke_log" >>"${SMOKE_SUMMARY}"
        echo "[INFO] OOM at bsz=${candidate_bsz}; stopping search."
        break
    fi

    printf "%s\tfailed\t%s\t%s\n" "$candidate_bsz" "$smoke_status" "$smoke_log" >>"${SMOKE_SUMMARY}"
    echo "[ERROR] Batch-size smoke failed for a non-OOM reason at bsz=${candidate_bsz}."
    echo "[ERROR] See ${smoke_log}"
    exit "$smoke_status"
done

if [[ "$selected_batch_size" -le 0 ]]; then
    echo "[ERROR] No valid batch size found. Even bsz=64 failed under trades."
    echo "[ERROR] See ${SMOKE_SUMMARY}"
    exit 1
fi

BATCH_SIZE="$selected_batch_size"
echo "[INFO] Selected max batch size for ${arch}: ${BATCH_SIZE} (trades-safe)"
echo "[INFO] Batch-size smoke summary: ${SMOKE_SUMMARY}"

# ---- 2+3. protocol x compile timing probes (5 warmup + 20 measured batches) ----
overall_status=0
for protocol in "${PROTOCOLS[@]}"; do
    for compile_flag in False True; do
        probe_dir="${RUN_DIR}/probe_${protocol}_compile_${compile_flag}"
        mkdir -p "${probe_dir}"
        build_runtime_cmd "$protocol" "$compile_flag" "$BATCH_SIZE" "$probe_dir" "$CSV_PATH" 5 20
        echo "[INFO] Timing probe: arch=${arch} protocol=${protocol} compile=${compile_flag} bsz=${BATCH_SIZE}"
        "${cmd[@]}" >"${probe_dir}/probe.log" 2>&1
        status=$?
        if [[ "$status" -ne 0 ]]; then
            echo "[ERROR] Probe failed (arch=${arch} protocol=${protocol} compile=${compile_flag}), exit=${status}"
            echo "[ERROR] See ${probe_dir}/probe.log"
            overall_status="$status"
        fi
    done
done

echo "[INFO] Done. Results CSV: ${CSV_PATH}"
exit "$overall_status"
