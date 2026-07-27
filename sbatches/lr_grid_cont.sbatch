#!/bin/bash
#SBATCH --job-name=cnb_l2_c46_lrgrid
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/lr_grid_cont/%j.out

# Self-contained single-run continuation launcher for the convnext_base L2 4->6
# learning-rate grid search. Does NOT use the cont_train name-parser, so RUN_NAME
# is free-form and the three grid runs never collide.
#
# Required (via --export): RUN_NAME
# Optional: LR (exact peak optimizer.lr; unset = let the training script derive it
#           from lr_scheduler.lrb * batch_size / 512), NORM (l2|linf, default l2),
#           CKPT_NAME (source checkpoint file, default model_best.pth.tar),
#           SOURCE_MODEL (default convnext_base_${NORM}_4_init0), BATCH (default 128),
#           EPOCHS (default 40), ATTACK_IT (PGD steps, default unset = use the
#           training config's default)

set -euo pipefail
nvidia-smi

# Reuse conda-env selection helper (by GPU memory) from the shared launcher.
source /home/ashtomer/projects/ares/sbatches/cont_train_launcher.sh

GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
echo "[INFO] Detected GPU Memory: ${GPU_TOTAL} MB"
module load anaconda
source activate "$(select_cont_train_env "$GPU_TOTAL")"
cd /home/ashtomer/projects/ares

: "${RUN_NAME:?set RUN_NAME (e.g. convnext_base_l2_cont4to6_lr1e4_init0)}"
LR="${LR:-}"
NORM="${NORM:-l2}"
CKPT_NAME="${CKPT_NAME:-model_best.pth.tar}"
SOURCE_MODEL="${SOURCE_MODEL:-convnext_base_${NORM}_4_init0}"
MODELS_ROOT=/home/ashtomer/projects/ares/results/models
CKPT="${MODELS_ROOT}/${SOURCE_MODEL}/${CKPT_NAME}"
[[ -f "$CKPT" ]] || { echo "[ERROR] missing source ckpt: $CKPT"; exit 1; }
BATCH="${BATCH:-128}"
EPOCHS="${EPOCHS:-40}"
ATTACK_IT="${ATTACK_IT:-}"

echo "RUN_NAME    : $RUN_NAME"
echo "attack_norm : $NORM"
echo "optimizer.lr: ${LR:-<auto: lrb * batch_size / 512>}"
echo "batch_size  : $BATCH"
echo "epochs      : $EPOCHS"
echo "attack_it   : ${ATTACK_IT:-<default>}"
echo "checkpoint  : $CKPT"

ATTACK_IT_OVERRIDE=()
[[ -n "$ATTACK_IT" ]] && ATTACK_IT_OVERRIDE=(attacks.attack_it="$ATTACK_IT")
LR_OVERRIDE=()
[[ -n "$LR" ]] && LR_OVERRIDE=(optimizer.lr="$LR")

python -m robust_training.adversarial_training \
  model=convnext_base \
  attacks.attack_norm="$NORM" attacks.advtrain=true attacks.gradnorm=false \
  attacks.attack_criterion=madry attacks.attack_domain=pixel attacks.attack_eps=6 \
  "${ATTACK_IT_OVERRIDE[@]}" \
  continuation.enabled=true continuation.use_ema=true continuation.checkpoint_path="$CKPT" \
  epsilon_schedule.enabled=true epsilon_schedule.type=warmup_ramp_fixed \
  epsilon_schedule.source_epsilon=4 epsilon_schedule.target_epsilon=6 \
  epsilon_schedule.warmup_epochs=4 epsilon_schedule.ramp_start_epoch=5 \
  epsilon_schedule.ramp_end_epoch=30 epsilon_schedule.fixed_start_epoch=31 \
  checkpointing.save_best_adv=true \
  "${LR_OVERRIDE[@]}" \
  model.experiment_name="$RUN_NAME" model.experiment_num=0 model.compile_model=True \
  model.resume="${MODELS_ROOT}/${RUN_NAME}/last.pth.tar" \
  training.epochs="$EPOCHS" training.batch_size="$BATCH" \
  hydra.run.dir="${MODELS_ROOT}/${RUN_NAME}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
