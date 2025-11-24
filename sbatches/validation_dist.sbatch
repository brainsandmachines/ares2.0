#!/bin/bash
#SBATCH --job-name=validation_dist
#SBATCH --time=14-00:00:00
#SBATCH --partition=rtx6000
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/validation_distance.out

cd /home/ashtomer/projects/ares || exit 1

python /home/ashtomer/projects/ares/data_analysis/validation_set_analysis.py