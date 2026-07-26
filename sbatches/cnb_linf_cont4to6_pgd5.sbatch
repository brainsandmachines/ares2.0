#!/bin/bash
#SBATCH --job-name=cnb_linf_c46_pgd5
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/cnb_linf_cont4to6_pgd5/%j.out

# One-off convnext_base linf 4->6 continuation test: everything matches the
# AIRCC convnext_base_linf_cont4to6_init0 run that stayed stable (bsz=256
# instead of AIRCC's 512, no LR override, no grad clip, same ramp schedule
# ramp_end=25/fixed_start=26), except attack_it=5 instead of 3 PGD steps, to
# test whether a stronger inner attack avoids the catastrophic-overfitting
# collapse seen on the lr-grid runs (which used ramp_end=30/fixed_start=31
# and attack_it=3).

set -euo pipefail
nvidia-smi

source /home/ashtomer/projects/ares/sbatches/cont_train_launcher.sh

GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
echo "[INFO] Detected GPU Memory: ${GPU_TOTAL} MB"
module load anaconda
source activate "$(select_cont_train_env "$GPU_TOTAL")"
cd /home/ashtomer/projects/ares

RUN_NAME="convnext_base_linf_cont4to6_pgd5_init0"
MODELS_ROOT=/home/ashtomer/projects/ares/results/models
CKPT="${MODELS_ROOT}/convnext_base_linf_4_init0/last.pth.tar"
[[ -f "$CKPT" ]] || { echo "[ERROR] missing source ckpt: $CKPT"; exit 1; }
BATCH=256
EPOCHS=40
ATTACK_IT=5

echo "RUN_NAME    : $RUN_NAME"
echo "attack_norm : linf"
echo "attack_it   : $ATTACK_IT"
echo "batch_size  : $BATCH"
echo "epochs      : $EPOCHS"
echo "checkpoint  : $CKPT"

python -m robust_training.adversarial_training \
  model=convnext_base \
  attacks.attack_norm=linf attacks.advtrain=true attacks.gradnorm=false \
  attacks.attack_criterion=madry attacks.attack_domain=pixel attacks.attack_eps=6 \
  attacks.attack_it="$ATTACK_IT" \
  continuation.enabled=true continuation.use_ema=true continuation.checkpoint_path="$CKPT" \
  epsilon_schedule.enabled=true epsilon_schedule.type=warmup_ramp_fixed \
  epsilon_schedule.source_epsilon=4 epsilon_schedule.target_epsilon=6 \
  epsilon_schedule.warmup_epochs=4 epsilon_schedule.ramp_start_epoch=5 \
  epsilon_schedule.ramp_end_epoch=25 epsilon_schedule.fixed_start_epoch=26 \
  checkpointing.save_best_adv=true \
  model.experiment_name="$RUN_NAME" model.experiment_num=0 model.compile_model=True \
  model.resume="${MODELS_ROOT}/${RUN_NAME}/last.pth.tar" \
  training.epochs="$EPOCHS" training.batch_size="$BATCH" \
  hydra.run.dir="${MODELS_ROOT}/${RUN_NAME}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
