#!/bin/bash
# Build and run ONE orchestrated stage (train or AA) for the claimed model.
#
# Invoked by orchestrator.controller:
#     bash sbatches/run_stage.sh <train|aa>
#
# Required env (exported by the controller / controller.sbatch):
#     ORCH_MODEL_ID, ORCH_DB          - orchestrator hooks write status/epoch
#     SLURM_JOB_NAME                  - the model's job name (drives parse_train_job)
#     ORCH_TARGET_EPOCH               - authoritative run length (training.epochs)
#
# Command construction mirrors sbatches/golan-trainmodels.sbatch exactly (it
# reuses train_launcher_lib.sh::parse_train_job), then appends mode-specific
# flags. Stage handoffs (training -> AA_EVAL, AA -> PLOTTING) are written by the
# in-job Python hooks, not here.
set -o pipefail

MODE="${1:?usage: run_stage.sh <train|aa>}"
REPO="${ORCH_CLUSTER_REPO:-/home/ashtomer/projects/ares}"
cd "$REPO" || exit 1

source "$REPO/sbatches/train_launcher_lib.sh"

GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[run_stage] mode=$MODE model=$ORCH_MODEL_ID job=$SLURM_JOB_NAME gpu=${GPU_TOTAL}MB target=${ORCH_TARGET_EPOCH}"
module load anaconda
source activate "$(select_train_env "$GPU_TOTAL")"
export MASTER_PORT=$((10000 + RANDOM % 50000))

jobname="$SLURM_JOB_NAME"
parse_train_job "$jobname" "$GPU_TOTAL" || exit 1

MODELS_ROOT="${MODELS_ROOT:-$REPO/results/models}"
MODEL_NAME_SUFFIX="${MODEL_NAME_SUFFIX:-}"
model_name="${model_name}${MODEL_NAME_SUFFIX}"

LRB=1.0e-3
if [[ "$attack_eps" == "8" ]]; then
    LRB=5.0e-4
elif [[ "$attack_eps" == "16" ]]; then
    LRB=2.5e-4
fi
LRB="${TRAIN_LRB:-$LRB}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-$BATCH_SIZE}"
INIT_FROM_MODEL="${INIT_FROM_MODEL:-}"
INIT_FROM_CKPT="${INIT_FROM_CKPT:-model_best.pth.tar}"
GRADNORM_OBJECTIVE="${GRADNORM_OBJECTIVE:-}"
GRADNORM_MAX_REG_TO_CE_RATIO="${GRADNORM_MAX_REG_TO_CE_RATIO:-}"

# Resume from the rolling checkpoint (auto): training skips the loop once the
# checkpoint is at ORCH_TARGET_EPOCH, so the AA stage runs eval-only.
RESUME_PATH="${MODELS_ROOT}/${model_name}/last.pth.tar"

if [[ "$dvd_enabled" == "true" ]]; then
    python - <<'PY'
import importlib
for name in ("kornia", "dvd"):
    importlib.import_module(name)
print("[INFO] DVD dependencies available")
PY
    if [[ "$?" -ne 0 ]]; then
        echo "[ERROR] DVD dependencies are missing from the selected training environment" >&2
        exit 1
    fi
fi

cmd=(
    python -m robust_training.adversarial_training
    attacks.attack_norm="$attack_norm"
    attacks.advtrain="$advtrain"
    attacks.gradnorm="$gradnorm"
    attacks.gradnorm_penalty_norm="$gradnorm_penalty_norm"
    attacks.attack_criterion="$crit"
    attacks.attack_domain="$attack_domain"
    model.experiment_num="$experiment_num"
    model.experiment_name="$model_name"
    model.resume="$RESUME_PATH"
    training.batch_size=$BATCH_SIZE
    training.epochs="$ORCH_TARGET_EPOCH"
    lr_scheduler.lrb="$LRB"
    # Match golan-trainmodels.sbatch exactly: ${now:...} expands to empty here, so
    # output_dir collapses to ${MODELS_ROOT}/${model_name}/ -- the flat dir that
    # RESUME_PATH and the AA/plot stages read from. Do NOT escape it (that would
    # create timestamped subdirs and break resume).
    hydra.run.dir="${MODELS_ROOT}/$model_name/${now:%Y-%m-%d}/${now:%H-%M-%S}"
)

if [[ -n "$INIT_FROM_MODEL" ]]; then
    cmd+=(
        "continuation.enabled=true"
        "continuation.checkpoint_path=${MODELS_ROOT}/${INIT_FROM_MODEL}/${INIT_FROM_CKPT}"
        "continuation.use_ema=true"
    )
fi
if [[ -n "$GRADNORM_OBJECTIVE" ]]; then
    cmd+=("attacks.gradnorm_objective=$GRADNORM_OBJECTIVE")
fi
if [[ -n "$GRADNORM_MAX_REG_TO_CE_RATIO" ]]; then
    cmd+=("attacks.gradnorm_max_reg_to_ce_ratio=$GRADNORM_MAX_REG_TO_CE_RATIO")
fi
if [[ "$dvd_enabled" == "true" ]]; then
    cmd+=("dataset.dvd.enabled=true" "dataset.dvd.variant=$dvd_variant")
fi
if [[ "$advtrain" == "true" ]] && {
      [[ "$crit" == "madry" && "$attack_domain" == "pixel" ]] ||
      [[ "$crit" == "trades" && "$attack_domain" == "v1_feature" && "$GPU_TOTAL" -gt 90000 ]]
  }; then
    cmd+=("model.compile_model=True")
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

# ---- mode-specific eval flags ----
if [[ "$MODE" == "train" ]]; then
    # Train only; the training-complete hook advances the row to AA_EVAL.
    cmd+=("final_eval=False")
elif [[ "$MODE" == "aa" ]]; then
    # Eval-only at max epoch; the AA hook advances the row to PLOTTING.
    cmd+=("final_eval=True" "final_eval_autoattack=True" "final_eval_pgd=False")
else
    echo "[run_stage] unknown mode: $MODE" >&2
    exit 2
fi

echo "[run_stage] exec: ${cmd[*]}"
exec "${cmd[@]}"
