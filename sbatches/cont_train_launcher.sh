#!/bin/bash

select_cont_train_env() {
    local gpu_total="$1"
    if [ "$gpu_total" -ge 90000 ]; then
        echo "tomer_advtrain_pro"
    else
        echo "tomer_advtrain"
    fi
}

parse_cont_train_job() {
    local jobname="$1"
    local gpu_total="$2"
    local models_root="${3:-/home/ashtomer/projects/ares/results/models}"

    model_size="small"
    job_core="$jobname"
    if [[ "$jobname" == convnext_base_* ]]; then
        model_size="base"
        job_core="${jobname#convnext_base_}"
    elif [[ "$jobname" == convnext_large_* ]]; then
        model_size="large"
        job_core="${jobname#convnext_large_}"
    fi

    if [[ ! "$job_core" =~ ^(.+)_cont([0-9]+(\.[0-9]+)?)to([0-9]+(\.[0-9]+)?)_(direct|ramp)_init([0-9]+)$ ]]; then
        echo "[ERROR] Continuation job must look like <mode>_cont<SRC>to<TGT>_<direct|ramp>_init<N>: $jobname" >&2
        return 1
    fi

    cont_mode="${BASH_REMATCH[1]}"
    source_eps="${BASH_REMATCH[2]}"
    target_eps="${BASH_REMATCH[4]}"
    protocol="${BASH_REMATCH[6]}"
    experiment_num="${BASH_REMATCH[7]}"

    attack_norm=""
    crit="madry"
    advtrain=true
    gradnorm=false
    gradnorm_penalty_norm="l1"
    is_v1=false
    v1_noise_mode=""
    attack_domain="pixel"
    model_override=""

    local mode_core="$cont_mode"
    local v1_tag=""
    if [[ "$mode_core" == v1clean_* ]]; then
        is_v1=true
        v1_tag="clean"
        v1_noise_mode="null"
        mode_core="${mode_core#v1clean_}"
    elif [[ "$mode_core" == v1noise_* ]]; then
        echo "[ERROR] V1 noise adversarial continuation is not supported: $jobname" >&2
        return 1
    fi

    if [[ "$mode_core" == *gradnorm* ]]; then
        echo "[ERROR] GradNorm continuation is not supported by this launcher: $jobname" >&2
        return 1
    fi

    if [[ "$mode_core" == *trades* ]]; then
        crit="trades"
    fi

    if [[ "$mode_core" == linf* ]]; then
        attack_norm="linf"
    elif [[ "$mode_core" == l2* ]]; then
        attack_norm="l2"
    elif [[ "$mode_core" == l1* ]]; then
        attack_norm="l1"
    else
        echo "[ERROR] Could not infer attack norm from continuation mode: $mode_core" >&2
        return 1
    fi

    local model_arch="convnext_${model_size}"
    if [[ "$is_v1" == "true" ]]; then
        attack_domain="v1_feature"
        model_override="model=${model_arch}_v1"
        source_model_name="${model_arch}_v1_${v1_tag}_${mode_core}_${source_eps}_init${experiment_num}"
        model_name="${model_arch}_v1_${v1_tag}_${mode_core}_cont${source_eps}to${target_eps}_${protocol}_init${experiment_num}"
    else
        if [[ "$model_size" != "small" ]]; then
            model_override="model=${model_arch}"
        fi
        source_model_name="${model_arch}_${mode_core}_${source_eps}_init${experiment_num}"
        model_name="${model_arch}_${mode_core}_cont${source_eps}to${target_eps}_${protocol}_init${experiment_num}"
    fi

    if [ "$gpu_total" -ge 90000 ]; then
        if [ "$crit" = "trades" ]; then
            BATCH_PER_GPU=384
        else
            BATCH_PER_GPU=512
        fi
    elif [ "$gpu_total" -ge 45000 ]; then
        if [ "$crit" = "trades" ]; then
            BATCH_PER_GPU=192
        else
            BATCH_PER_GPU=256
        fi
    elif [ "$gpu_total" -ge 22000 ]; then
        if [ "$crit" = "trades" ]; then
            BATCH_PER_GPU=96
        else
            BATCH_PER_GPU=128
        fi
    else
        echo "[ERROR] GPU total memory too small (< 22 GB). Detected: $gpu_total" >&2
        return 1
    fi

    if [[ "$model_size" == "large" ]]; then
        BATCH_PER_GPU=$((BATCH_PER_GPU * 3 / 4))
    fi
    BATCH_SIZE=$BATCH_PER_GPU

    if [[ "$protocol" == "direct" ]]; then
        schedule_type="fixed"
        epochs=30
        warmup_epochs=2
        ramp_start_epoch=""
        ramp_end_epoch=""
        fixed_start_epoch=""
    elif [[ "$protocol" == "ramp" ]]; then
        schedule_type="warmup_ramp_fixed"
        epochs=35
        warmup_epochs=3
        ramp_start_epoch=4
        ramp_end_epoch=20
        fixed_start_epoch=21
        epochs="${CONT_EPOCHS:-$epochs}"
        warmup_epochs="${CONT_WARMUP_EPOCHS:-$warmup_epochs}"
        ramp_start_epoch="${CONT_RAMP_START_EPOCH:-$ramp_start_epoch}"
        ramp_end_epoch="${CONT_RAMP_END_EPOCH:-$ramp_end_epoch}"
        fixed_start_epoch="${CONT_FIXED_START_EPOCH:-$fixed_start_epoch}"
    else
        echo "[ERROR] Unsupported continuation protocol: $protocol" >&2
        return 1
    fi

    if [[ -z "${CONTINUATION_CKPT:-}" ]]; then
        local adv_ckpt="${models_root}/${source_model_name}/model_best_adv.pth.tar"
        local best_ckpt="${models_root}/${source_model_name}/model_best.pth.tar"
        if [[ -f "$adv_ckpt" ]]; then
            CONTINUATION_CKPT="$adv_ckpt"
        elif [[ -f "$best_ckpt" ]]; then
            CONTINUATION_CKPT="$best_ckpt"
        else
            echo "[ERROR] No source checkpoint found for ${source_model_name}" >&2
            echo "[ERROR] Tried: $adv_ckpt" >&2
            echo "[ERROR] Tried: $best_ckpt" >&2
            echo "[ERROR] Set CONTINUATION_CKPT=/path/to/checkpoint.pth.tar to override." >&2
            return 1
        fi
    fi

    if [[ ! -f "$CONTINUATION_CKPT" ]]; then
        echo "[ERROR] CONTINUATION_CKPT does not exist: $CONTINUATION_CKPT" >&2
        return 1
    fi
}
