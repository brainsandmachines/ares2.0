#!/bin/bash
#SBATCH --job-name=l2_16_init1
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --array=1-10%1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/golan_neuro/convnextsmall/%A/%a.out



nvidia-smi
# -----------------------------
# 1. Get GPU Memory (in MB)
# -----------------------------
# This extracts the number (e.g., "48000") from nvidia-smi
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[INFO] Detected GPU Memory: ${GPU_TOTAL} MB"
# ---- Env once ----
if [ "$GPU_TOTAL" -ge 90000 ]; then
    module load anaconda
    source activate tomer_advtrain_pro
else
    module load anaconda
    source activate tomer_advtrain  
fi
# ------------------


cd /home/ashtomer/projects/ares

python -c "import torch; print(torch.version.cuda, torch.version.git_version)"

export MASTER_PORT=$((10000 + RANDOM % 50000))

jobname="$SLURM_JOB_NAME"

# -----------------------------
# Parse experiment_num (number after init)
# -----------------------------
attack_eps=""
experiment_num=""
if [[ "$jobname" =~ _([0-9]+)_init([0-9]+)$ ]]; then
    attack_eps="${BASH_REMATCH[1]}"
    experiment_num="${BASH_REMATCH[2]}"
elif [[ "$jobname" =~ _init([0-9]+)$ ]]; then
    experiment_num="${BASH_REMATCH[1]}"
else
    echo "[ERROR] Could not parse experiment_num from job name: $jobname"
    exit 1
fi

# -----------------------------
# Defaults
# -----------------------------
attack_norm="linf"
advtrain=false
gradnorm=false
gradnorm_penalty_norm="l1"
crit="madry"
is_v1=false
v1_noise_mode=""
attack_domain="pixel"
model_override=""

# -----------------------------
# -----------------------------
# Rules based on job name
# -----------------------------
if [[ "$jobname" == v1clean* ]]; then
    is_v1=true
    v1_noise_mode="null"
    model_override="model=convnext_small_v1"
elif [[ "$jobname" == v1noise* ]]; then
    is_v1=true
    v1_noise_mode="neuronal"
    model_override="model=convnext_small_v1"
fi

if [[ "$jobname" == *gradnorm* ]]; then
    gradnorm=true
    if [[ "$jobname" == *gradnorm_l2* ]]; then
        gradnorm_penalty_norm="l2"
    fi
elif [[ "$jobname" == *linf* ]]; then
    attack_norm="linf"
    advtrain=true
elif [[ "$jobname" == *l2* ]]; then
    attack_norm="l2"
    advtrain=true
elif [[ "$jobname" == *l1* ]]; then
    attack_norm="l1"
    advtrain=true
fi

if [[ "$jobname" == *trades* ]]; then
    crit="trades"
fi

if [[ "$gradnorm" == "true" ]]; then
    if [[ "$jobname" != *gradnorm_l1* && "$jobname" != *gradnorm_l2* ]]; then
        echo "[ERROR] GradNorm job names must include gradnorm_l1 or gradnorm_l2: $jobname"
        exit 1
    fi
fi

if [[ "$advtrain" == "true" || "$gradnorm" == "true" ]]; then
    if [[ -z "$attack_eps" ]]; then
        echo "[ERROR] Could not parse attack epsilon from job name: $jobname"
        exit 1
    fi
fi

if [[ "$is_v1" == "true" && "$gradnorm" == "true" ]]; then
    echo "[ERROR] V1 job names do not support gradnorm: $jobname"
    exit 1
fi

if [[ "$is_v1" == "true" && "$advtrain" == "true" ]]; then
    attack_domain="v1_feature"
    if [[ "$v1_noise_mode" != "null" ]]; then
        echo "[ERROR] V1 adversarial training with noise is not supported: $jobname"
        exit 1
    fi
fi



# -----------------------------
# 2. Set Batch Size
# -----------------------------
if [ "$GPU_TOTAL" -ge 90000 ]; then
    # A100 (80GB) range
    if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
        BATCH_PER_GPU=384
    else
        BATCH_PER_GPU=512
    fi
elif [ "$GPU_TOTAL" -ge 45000 ]; then
    # RTX 6000 / A6000 (48GB) range
    if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
        BATCH_PER_GPU=192
    else
        BATCH_PER_GPU=256
    fi
elif [ "$GPU_TOTAL" -ge 22000 ]; then
    # 3090 / 4090 / A10 (24GB) range
    if [ "$gradnorm" = "true" ] || [ "$crit" = "trades" ]; then
        BATCH_PER_GPU=96
    else
        BATCH_PER_GPU=128
    fi
else
    echo "[ERROR] GPU total memory too small (< 22 GB). Detected: $GPU_TOTAL"
    exit 1
fi

BATCH_SIZE=$BATCH_PER_GPU

# -----------------------------
# Model name
# -----------------------------

if [[ "$is_v1" == "true" ]]; then
      v1_tag="clean"
      if [[ "$v1_noise_mode" != "null" ]]; then
          v1_tag="noise"
      fi

      if [[ "$advtrain" == "true" ]]; then
          if [[ "$crit" == "trades" ]]; then
              if [[ "$attack_norm" == "linf" ]]; then
                  model_name="convnext_small_v1_${v1_tag}_linftrades_${attack_eps}_init${experiment_num}"
              elif [[ "$attack_norm" == "l2" ]]; then
                  model_name="convnext_small_v1_${v1_tag}_l2trades_${attack_eps}_init${experiment_num}"
              elif [[ "$attack_norm" == "l1" ]]; then
                  model_name="convnext_small_v1_${v1_tag}_l1trades_${attack_eps}_init${experiment_num}"
              else
                  echo "[ERROR] Unsupported attack norm for V1 TRADES: $attack_norm"
                  exit 1
              fi
          else
              if [[ "$attack_norm" == "linf" ]]; then
                  model_name="convnext_small_v1_${v1_tag}_linf_${attack_eps}_init${experiment_num}"
              elif [[ "$attack_norm" == "l2" ]]; then
                  model_name="convnext_small_v1_${v1_tag}_l2_${attack_eps}_init${experiment_num}"
              elif [[ "$attack_norm" == "l1" ]]; then
                  model_name="convnext_small_v1_${v1_tag}_l1_${attack_eps}_init${experiment_num}"
              else
                  echo "[ERROR] Unsupported attack norm for V1 Madry: $attack_norm"
                  exit 1
              fi
          fi
      else
          model_name="convnext_small_v1_${v1_tag}_init${experiment_num}"
      fi
  else
      model_name="convnext_small_${jobname}"
  fi


echo "JOB NAME        : $jobname"
echo "attack_eps     : $attack_eps"
echo "experiment_num : $experiment_num"
echo "attack_norm    : $attack_norm"
echo "advtrain       : $advtrain"
echo "gradnorm       : $gradnorm"
echo "gradnorm_pen   : $gradnorm_penalty_norm"
echo "is_v1          : $is_v1"
echo "v1_noise_mode  : $v1_noise_mode"
echo "attack_domain  : $attack_domain"
echo "model_name     : $model_name"
echo "Batch size     : $BATCH_SIZE"
echo "attack_crit    : $crit"

cmd=(
    python -m robust_training.adversarial_training
    attacks.attack_norm="$attack_norm"
    attacks.advtrain="$advtrain"
    attacks.gradnorm="$gradnorm"
    attacks.gradnorm_penalty_norm="$gradnorm_penalty_norm"
    attacks.attack_criterion="$crit"
    attacks.attack_domain="$attack_domain"
    model.experiment_num="$experiment_num"
    model.resume="/home/ashtomer/projects/ares/results/models/$model_name/last.pth.tar"
    training.batch_size=$BATCH_SIZE
    hydra.run.dir="/home/ashtomer/projects/ares/results/models/$model_name/${now:%Y-%m-%d}/${now:%H-%M-%S}"
)

if [[ -n "$model_override" ]]; then
    cmd+=("$model_override")
    cmd+=("model.v1_noise_mode=$v1_noise_mode")
fi

if [[ "$is_v1" == "true" && "$advtrain" == "true" ]]; then
    cmd+=("attacks.v1_attack_eps=$attack_eps")
elif [[ -n "$attack_eps" ]]; then
    cmd+=("attacks.attack_eps=$attack_eps")
fi

"${cmd[@]}"

status=$?
echo "[INFO] Training exit status: $status"

if [ "$status" -eq 0 ]; then
    echo "[INFO] Running checkpoint pruning for $model_name"
    python /home/ashtomer/projects/ares/results/models/prune_model_checkpoints.py \
        "/home/ashtomer/projects/ares/results/models/$model_name" \
        --keep-n 10
    echo "[INFO] Checkpoint pruning finished with status $?"
else
    echo "[INFO] Skipping checkpoint pruning because training failed or was killed"
fi

exit "$status"