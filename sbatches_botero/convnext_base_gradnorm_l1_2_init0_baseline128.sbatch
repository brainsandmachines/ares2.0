#!/bin/bash
#SBATCH --job-name=convnext_base_gradnorm_l1_2_init0_baseline128
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/golan_neuro/convnextbase_gradnorm/%j.out

nvidia-smi
module load anaconda
source activate tomer_advtrain

cd /home/ashtomer/projects/ares

export HYDRA_FULL_ERROR=1

python -m robust_training.adversarial_training \
    model=convnext_base \
    attacks.attack_norm=linf \
    attacks.attack_eps=2 \
    attacks.advtrain=false \
    attacks.gradnorm=true \
    attacks.gradnorm_penalty_norm=l1 \
    attacks.attack_criterion=madry \
    attacks.attack_domain=pixel \
    optimizer.clip_grad=1.0 \
    optimizer.clip_mode=norm \
    model.experiment_num=0 \
    model.experiment_name=convnext_base_gradnorm_l1_2_init0_from_baseline128 \
    model.resume=/home/ashtomer/projects/ares/results/models/convnext_base_baseline_init0/model_best.pth.tar \
    training.epochs=128 \
    training.batch_size=128 \
    checkpointing.save_best_adv=true \
    final_eval=true \
    final_eval_autoattack=true \
    final_eval_pgd=false \
    output_dir=results/models
