#!/bin/bash
# Sequential runtime epoch inspection for AIRCC, 2 parallel Python procs per trial.
# Replicates runtime_epoch_inspection.sbatch inside the ares-train:v1 container.
#
# Default: convnext_small only (~90 min in sandbox).
# PROBE_ARCHS="small base large" to run all three (space-separated).

set -uo pipefail

REPO_ROOT="/shared/cycle2_bgu_golan_prj/ashtomer/ares"
cd "${REPO_ROOT}"

# Source parse_train_job — pure bash functions, no conda/module side effects
source "${REPO_ROOT}/sbatches/train_launcher_lib.sh"

export WANDB_MODE=disabled
export MASTER_PORT=$((10000 + RANDOM % 50000))

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

# Dataset: use whatever is available in imagenet1k; fall back to synthetic tiny dataset
IMAGENET_DIR="/shared/cycle2_bgu_golan_prj/datasets/imagenet1k"
TINY_DIR="/shared/cycle2_bgu_golan_prj/datasets/imagenet_tiny"
N_TRAIN=$(find "${IMAGENET_DIR}/train" -name "*.JPEG" 2>/dev/null | head -1 | wc -l)
if [[ $N_TRAIN -gt 0 ]]; then
    TRAIN_DIR="${IMAGENET_DIR}/train"
    EVAL_DIR="${IMAGENET_DIR}/val"
    echo "[INFO] Using imagenet1k (partial rsync OK)"
else
    TRAIN_DIR="${TINY_DIR}/train"
    EVAL_DIR="${TINY_DIR}/val"
    echo "[INFO] imagenet1k has no images yet — using imagenet_tiny fallback"
fi
echo "[INFO] TRAIN_DIR=${TRAIN_DIR}"

GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[INFO] GPU Memory: ${GPU_TOTAL} MB"

# 6 protocol families — one representative norm per family
PROTOCOLS=(linf_8_init1  linftrades_8_init1  gradnorm_l1_8_init1  dvd_b_init1  v1clean_l2_8_init1  baseline_init1)
PLABELS=(  madry          trades               gradnorm              dvd           v1_l2                baseline)
ARCHS=(small base large)
PREFIX=("" convnext_base_ convnext_large_)

IFS=' ' read -ra PROBE_ARCHS_ARR <<< "${PROBE_ARCHS:-small}"

CSV_PATH="${REPO_ROOT}/research/runtime-arch-eval/aircc_runtime_arch_eval.csv"
DL_CSV="${REPO_ROOT}/research/runtime-arch-eval/aircc_dataloader_bench.csv"
RUN_DIR="${REPO_ROOT}/outs/advtrain/runtime_arch_eval_aircc/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}" "$(dirname "${CSV_PATH}")"

is_oom_log() { grep -Eiq 'out of memory|CUDA out of memory|DefaultCPUAllocator' "$1"; }

build_cmd() {
    local bsz=$1 out_dir=$2 csv=$3 warmup=$4 measure=$5 proc_id=$6
    cmd=(
        python -m robust_training.adversarial_training
        +machine=aircc
        attacks.attack_norm="${attack_norm}"
        attacks.advtrain="${advtrain}"
        attacks.gradnorm="${gradnorm}"
        attacks.gradnorm_penalty_norm="${gradnorm_penalty_norm}"
        attacks.attack_criterion="${crit}"
        attacks.attack_domain="${attack_domain}"
        model.experiment_num="${experiment_num}"
        model.experiment_name="runtime_probe_${protocol_label}_${arch_label}_p${proc_id}"
        model.resume=""
        training.batch_size="${bsz}"
        training.epochs=1
        training.model_ema=false
        final_eval=false
        dataset.train_dir="${TRAIN_DIR}"
        dataset.eval_dir="${EVAL_DIR}"
        output_dir="${out_dir}"
        "hydra.run.dir=${out_dir}/hydra"
        runtime_probe=true
        "runtime_probe_csv=${csv}"
        runtime_probe_protocol="${protocol_label}"
        runtime_probe_arch="${arch_label}"
        runtime_probe_warmup_batches="${warmup}"
        runtime_probe_measure_batches="${measure}"
    )
    [[ -n "${model_override:-}" ]] && cmd+=("${model_override}")
    [[ "${is_v1:-false}" == "true" ]] && cmd+=("model.v1_noise_mode=${v1_noise_mode:-null}")
    [[ "${compile_model:-false}" == "true" ]] && cmd+=("model.compile_model=True")
    [[ "${is_v1:-false}" == "true" && "${advtrain}" == "true" ]] && cmd+=("attacks.v1_attack_eps=${attack_eps}")
    [[ "${is_v1:-false}" != "true" && -n "${attack_eps:-}" ]] && cmd+=("attacks.attack_eps=${attack_eps}")
}

# Launch 2 parallel processes; returns 0 if both succeed, 2 if OOM detected, 1 otherwise.
run_parallel() {
    local dir=$1 csv=$2 warmup=$3 measure=$4 bsz=$5
    mkdir -p "${dir}/proc_0" "${dir}/proc_1"

    build_cmd "${bsz}" "${dir}/proc_0" "${csv}" "${warmup}" "${measure}" 0
    MASTER_PORT=${MASTER_PORT} "${cmd[@]}" > "${dir}/proc_0.log" 2>&1 &
    PID0=$!

    build_cmd "${bsz}" "${dir}/proc_1" "${csv}" "${warmup}" "${measure}" 1
    MASTER_PORT=$((MASTER_PORT + 1)) "${cmd[@]}" > "${dir}/proc_1.log" 2>&1 &
    PID1=$!

    wait $PID0; S0=$?
    wait $PID1; S1=$?

    if is_oom_log "${dir}/proc_0.log" || is_oom_log "${dir}/proc_1.log"; then
        return 2
    fi
    [[ $S0 -eq 0 && $S1 -eq 0 ]] && return 0 || return 1
}

declare -A dl_bench_done

task_id=0
for p_idx in "${!PROTOCOLS[@]}"; do
  for a_idx in "${!ARCHS[@]}"; do
    arch_label="${ARCHS[$a_idx]}"
    [[ " ${PROBE_ARCHS_ARR[*]} " != *" ${arch_label} "* ]] && { ((task_id++)); continue; }

    protocol_label="${PLABELS[$p_idx]}"
    jobname="${PREFIX[$a_idx]}${PROTOCOLS[$p_idx]}"
    echo; echo "==== [TASK ${task_id}] ${protocol_label} / ${arch_label} ===="

    parse_train_job "$jobname" "$GPU_TOTAL" || {
        echo "[ERROR] parse_train_job failed for ${jobname}"; ((task_id++)); continue
    }

    # Compile only for madry + pixel-domain attacks (matches production policy)
    compile_model=false
    [[ "${advtrain}" == "true" && "${crit}" == "madry" && "${attack_domain}" == "pixel" ]] && compile_model=true
    echo "[INFO] compile_model=${compile_model}"

    # DataLoader bench runs once per arch, before any smoke warms the page cache
    if [[ "${dl_bench_done[$arch_label]:-0}" == "0" ]]; then
        echo "[INFO] DataLoader num_workers sweep for arch=${arch_label} (bsz=256, 10 batches)"
        python "${REPO_ROOT}/aircc/dataloader_bench.py" \
            --train-dir "${TRAIN_DIR}" \
            --batch-size 256 \
            --num-batches 10 \
            --out-csv "${DL_CSV}" \
            --protocol "arch_${arch_label}" \
            --arch "${arch_label}"
        dl_bench_done[$arch_label]=1
    fi

    TASK_DIR="${RUN_DIR}/task_${task_id}_${protocol_label}_${arch_label}"
    SMOKE_ROOT="${TASK_DIR}/smoke"
    mkdir -p "${SMOKE_ROOT}"
    printf "bsz\tstatus\n" > "${SMOKE_ROOT}/summary.tsv"

    selected_bsz=0
    for candidate_bsz in 128 256 384 512; do
        smoke_dir="${SMOKE_ROOT}/bsz_${candidate_bsz}"
        echo "[INFO] Smoke (2x parallel) bsz=${candidate_bsz}"
        run_parallel "${smoke_dir}" "${smoke_dir}/smoke.csv" 0 1 "${candidate_bsz}"
        result=$?
        if [[ $result -eq 0 ]]; then
            selected_bsz=${candidate_bsz}
            printf "%s\tok\n" "$candidate_bsz" >> "${SMOKE_ROOT}/summary.tsv"
        elif [[ $result -eq 2 ]]; then
            printf "%s\toom\n" "$candidate_bsz" >> "${SMOKE_ROOT}/summary.tsv"
            echo "[INFO] OOM at bsz=${candidate_bsz} (2x parallel); max = ${selected_bsz}"; break
        else
            printf "%s\tfailed\n" "$candidate_bsz" >> "${SMOKE_ROOT}/summary.tsv"
            echo "[ERROR] Non-OOM failure at bsz=${candidate_bsz}"; break
        fi
    done

    if [[ $selected_bsz -le 0 ]]; then
        echo "[WARN] No valid bsz for ${protocol_label}/${arch_label}, skipping."
        ((task_id++)); continue
    fi

    echo "[INFO] Max bsz (2x parallel): ${selected_bsz}"
    echo "[INFO] Running timing probe (warmup=5, measure=20)"
    PROBE_DIR="${TASK_DIR}/probe_bsz_${selected_bsz}"
    run_parallel "${PROBE_DIR}" "${CSV_PATH}" 5 20 "${selected_bsz}"
    echo "[INFO] Probe done: exit=$?"
    ((task_id++))
  done
done

echo; echo "[DONE] Results → ${CSV_PATH}"
[[ -f "${CSV_PATH}" ]] && cat "${CSV_PATH}"
