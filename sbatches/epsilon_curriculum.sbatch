#!/bin/bash
#SBATCH --job-name=linf_cont4to8_init1
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx_pro_6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/epsilon_curriculum/%j.out

set -euo pipefail

nvidia-smi

source /home/ashtomer/projects/ares/sbatches/cont_train_launcher.sh
source /home/ashtomer/projects/ares/sbatches/stage_runner.sh

GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[INFO] Detected GPU Memory: ${GPU_TOTAL} MB"

module load anaconda
source activate "$(select_cont_train_env "$GPU_TOTAL")"

cd /home/ashtomer/projects/ares

python -c "import torch; print(torch.version.cuda, torch.version.git_version)"

export MASTER_PORT=$((10000 + RANDOM % 50000))

jobname="$SLURM_JOB_NAME"
MODELS_ROOT="${MODELS_ROOT:-/home/ashtomer/projects/ares/results/models}"

# Early bypass for PLOT_RESULTS / COMPLETED: skip parse_cont_train_job entirely.
if [[ -n "$ORCH_MODEL_ID" && -n "$ORCH_DB" ]]; then
    _stage="$(python -m orchestrator.jobctl get current_stage 2>/dev/null)"
    if [[ "$_stage" == "COMPLETED" || "$_stage" == "PLOT_RESULTS" ]]; then
        orch_pipeline  # COMPLETED → exit 0; PLOT_RESULTS → plot + advance COMPLETED
        exit $?
    fi
fi

parse_cont_train_job "$jobname" "$GPU_TOTAL" "$MODELS_ROOT" || exit 1

echo "JOB NAME        : $jobname"
echo "model_size      : $model_size"
echo "cont_mode       : $cont_mode"
echo "source_eps      : $source_eps"
echo "target_eps      : $target_eps"
echo "protocol        : $protocol"
echo "experiment_num  : $experiment_num"
echo "attack_norm     : $attack_norm"
echo "attack_crit     : $crit"
echo "attack_domain   : $attack_domain"
echo "is_v1           : $is_v1"
echo "model_name      : $model_name"
echo "source_model    : $source_model_name"
echo "checkpoint      : $CONTINUATION_CKPT"
echo "Batch size      : $BATCH_SIZE"

cmd=(
    python -m robust_training.adversarial_training
    attacks.attack_norm="$attack_norm"
    attacks.advtrain="$advtrain"
    attacks.gradnorm="$gradnorm"
    attacks.gradnorm_penalty_norm="$gradnorm_penalty_norm"
    attacks.attack_criterion="$crit"
    attacks.attack_domain="$attack_domain"
    continuation.enabled=true
    continuation.checkpoint_path="$CONTINUATION_CKPT"
    continuation.use_ema=true
    epsilon_schedule.enabled=true
    epsilon_schedule.type="$schedule_type"
    epsilon_schedule.source_epsilon="$source_eps"
    epsilon_schedule.target_epsilon="$target_eps"
    checkpointing.save_best_adv=true
    model.experiment_name="$model_name"
    model.experiment_num="$experiment_num"
    model.resume="${MODELS_ROOT}/${model_name}/last.pth.tar"
    training.epochs="$epochs"
    training.batch_size="$BATCH_SIZE"
    hydra.run.dir="${MODELS_ROOT}/${model_name}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
)

if [[ "$advtrain" == "true" && "$crit" == "madry" ]]; then
    cmd+=("model.compile_model=True")
    echo "compile_model  : True"
fi

cmd+=(
    epsilon_schedule.warmup_epochs="$warmup_epochs"
    epsilon_schedule.ramp_start_epoch="$ramp_start_epoch"
    epsilon_schedule.ramp_end_epoch="$ramp_end_epoch"
    epsilon_schedule.fixed_start_epoch="$fixed_start_epoch"
)

if [[ -n "$model_override" ]]; then
    cmd+=("$model_override")
    if [[ "$is_v1" == "true" ]]; then
        cmd+=("model.v1_noise_mode=$v1_noise_mode")
    fi
fi

if [[ "$is_v1" == "true" && "$advtrain" == "true" ]]; then
    cmd+=("attacks.v1_attack_eps=$target_eps")
elif [[ "$advtrain" == "true" ]]; then
    cmd+=("attacks.attack_eps=$target_eps")
fi

# Walk the orchestrator stage machine (TRAIN -> AA_SWEEP -> PLOT_RESULTS).
# Falls back to plain training when not orchestrated (ORCH_MODEL_ID unset).
orch_pipeline "${cmd[@]}"

status=$?
echo "[INFO] Pipeline exit status: $status"

if [ "$status" -eq 0 ]; then
    echo "[INFO] Running checkpoint pruning for $model_name"
    python /home/ashtomer/projects/ares/results/models/prune_model_checkpoints.py \
        "${MODELS_ROOT}/${model_name}" \
        --keep-n 10
    echo "[INFO] Checkpoint pruning finished with status $?"
else
    echo "[INFO] Skipping checkpoint pruning because training failed or was killed"
fi

exit "$status"
