#!/bin/bash
#SBATCH --job-name=convnext_small_training
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/advtrain/golan_neuro/convnextsmall/%A/%a.out
#SBATCH --array=1-500%5

cd /home/ashtomer/projects/ares/sh_files || exit 1

bash /home/ashtomer/projects/ares/sh_files/train_models.sh