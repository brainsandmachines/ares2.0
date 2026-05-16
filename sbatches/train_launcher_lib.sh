#!/bin/bash

select_train_env() {
    local gpu_total="$1"
    if [ "$gpu_total" -ge 90000 ]; then
        echo "tomer_advtrain_pro"
    else
        echo "tomer_advtrain"
    fi
}

parse_train_job() {
    local jobname="$1"
    local gpu_total="$2"

    model_size="small"
    job_core="$jobname"
    if [[ "$jobname" == convnext_base_* ]]; then
        model_size="base"
        job_core="${jobname#convnext_base_}"
    elif [[ "$jobname" == convnext_large_* ]]; then
        model_size="large"
        job_core="${jobname#convnext_large_}"
    fi

    attack_eps=""
    experiment_num=""
    if [[ "$job_core" =~ _([0-9]+)_init([0-9]+)$ ]]; then
        attack_eps="${BASH_REMATCH[1]}"
        experiment_num="${BASH_REMATCH[2]}"
    elif [[ "$job_core" =~ _init([0-9]+)$ ]]; then
        experiment_num="${BASH_REMATCH[1]}"
    else
        echo "[ERROR] Could not parse experiment_num from job name: $jobname" >&2
        return 1
    fi

    attack_norm="linf"
    advtrain=false
    gradnorm=false
    gradnorm_penalty_norm="l1"
    crit="madry"
    is_v1=false
    v1_noise_mode=""
    attack_domain="pixel"
    model_override=""

    local model_arch="convnext_${model_size}"

    if [[ "$job_core" == v1clean* ]]; then
        is_v1=true
        v1_noise_mode="null"
        model_override="model=${model_arch}_v1"
    elif [[ "$job_core" == v1noise* ]]; then
        is_v1=true
        v1_noise_mode="neuronal"
        model_override="model=${model_arch}_v1"
    elif [[ "$model_size" != "small" ]]; then
        model_override="model=${model_arch}"
    fi

    if [[ "$job_core" == *gradnorm* ]]; then
        gradnorm=true
        if [[ "$job_core" == *gradnorm_l2* ]]; then
            gradnorm_penalty_norm="l2"
        fi
    elif [[ "$job_core" == *linf* ]]; then
        attack_norm="linf"
        advtrain=true
    elif [[ "$job_core" == *l2* ]]; then
        attack_norm="l2"
        advtrain=true
    elif [[ "$job_core" == *l1* ]]; then
        attack_norm="l1"
        advtrain=true
    fi

    if [[ "$job_core" == *trades* ]]; then
        crit="trades"
    fi

    if [[ "$gradnorm" == "true" ]]; then
        if [[ "$job_core" != *gradnorm_l1* && "$job_core" != *gradnorm_l2* ]]; then
            echo "[ERROR] GradNorm job names must include gradnorm_l1 or gradnorm_l2: $jobname" >&2
            return 1
        fi
    fi

    if [[ "$advtrain" == "true" || "$gradnorm" == "true" ]]; then
        if [[ -z "$attack_eps" ]]; then
            echo "[ERROR] Could not parse attack epsilon from job name: $jobname" >&2
            return 1
        fi
    fi

    if [[ "$is_v1" == "true" && "$gradnorm" == "true" ]]; then
        echo "[ERROR] V1 job names do not support gradnorm: $jobname" >&2
        return 1
    fi

    if [[ "$is_v1" == "true" && "$advtrain" == "true" ]]; then
        attack_domain="v1_feature"
        if [[ "$v1_noise_mode" != "null" ]]; then
            echo "[ERROR] V1 adversarial training with noise is not supported: $jobname" >&2
            return 1
        fi
    fi

    if [ "$gpu_total" -ge 90000 ]; then
        if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
            BATCH_PER_GPU=384
        else
            BATCH_PER_GPU=512
        fi
    elif [ "$gpu_total" -ge 45000 ]; then
        if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
            BATCH_PER_GPU=192
        else
            BATCH_PER_GPU=256
        fi
    elif [ "$gpu_total" -ge 22000 ]; then
        if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
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

    if [[ "$is_v1" == "true" ]]; then
        local v1_tag="clean"
        if [[ "$v1_noise_mode" != "null" ]]; then
            v1_tag="noise"
        fi

        if [[ "$advtrain" == "true" ]]; then
            if [[ "$crit" == "trades" ]]; then
                if [[ "$attack_norm" == "linf" ]]; then
                    model_name="${model_arch}_v1_${v1_tag}_linftrades_${attack_eps}_init${experiment_num}"
                elif [[ "$attack_norm" == "l2" ]]; then
                    model_name="${model_arch}_v1_${v1_tag}_l2trades_${attack_eps}_init${experiment_num}"
                elif [[ "$attack_norm" == "l1" ]]; then
                    model_name="${model_arch}_v1_${v1_tag}_l1trades_${attack_eps}_init${experiment_num}"
                else
                    echo "[ERROR] Unsupported attack norm for V1 TRADES: $attack_norm" >&2
                    return 1
                fi
            else
                if [[ "$attack_norm" == "linf" ]]; then
                    model_name="${model_arch}_v1_${v1_tag}_linf_${attack_eps}_init${experiment_num}"
                elif [[ "$attack_norm" == "l2" ]]; then
                    model_name="${model_arch}_v1_${v1_tag}_l2_${attack_eps}_init${experiment_num}"
                elif [[ "$attack_norm" == "l1" ]]; then
                    model_name="${model_arch}_v1_${v1_tag}_l1_${attack_eps}_init${experiment_num}"
                else
                    echo "[ERROR] Unsupported attack norm for V1 Madry: $attack_norm" >&2
                    return 1
                fi
            fi
        else
            model_name="${model_arch}_v1_${v1_tag}_init${experiment_num}"
        fi
    else
        model_name="${model_arch}_${job_core}"
    fi
}
