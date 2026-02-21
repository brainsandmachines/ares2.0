#!/bin/bash
#SBATCH --job-name=convnext_small_training_test
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/golan_neuro/convnextsmall/%A/%a.out

nvidia-smi
# ---- Env once ----
module load anaconda
source activate tomer_advtrain
# ------------------

cd /home/ashtomer/projects/ares

export MASTER_PORT=$((10000 + RANDOM % 50000))
export HYDRA_FULL_ERROR=1

torchrun --nproc_per_node=1 --master-port=$MASTER_PORT -m robust_training.hydra_advtrain \
    attacks.attack_norm=linf \
    attacks.attack_eps=1 \
    attacks.advtrain=false \
    attacks.gradnorm=false \
    model.experiment_num=1 \
    model.experiment_name="convnext_small_baseline_test" \
    model.model_id="'baseline'" \
    training.epochs=250 \
    training.batch_size=256 \
    hydra.run.dir="/home/ashtomer/projects/ares/results/models/convnext_small/baseline" \
    output_dir="/home/ashtomer/projects/ares/results/models/convnext_small/baseline"