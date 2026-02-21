#!/bin/bash
#SBATCH --job-name=autoattack_eval
#SBATCH --time=7-00:00:00
#SBATCH --partition=rtx_3090                 # main queue
#SBATCH --gpus=1                          # request 1 GPU          
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/autoattack_eval/%A.out

nvidia-smi
# ---- Env once ----
module load anaconda
source activate tomer_advtrain
# ------------------


cd /home/ashtomer/projects/ares

python /home/ashtomer/projects/ares/data_analysis/autoattack_eval.py
