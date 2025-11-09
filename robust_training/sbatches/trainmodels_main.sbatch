#!/bin/bash
#SBATCH --job-name=convnext_small_training
#SBATCH --time=7-00:00:00
#SBATCH --partition=main                 # main queue
#SBATCH --gpus=1                          # request 1 GPU
#SBATCH --constraint=rtx_pro_6000            # GPUs with >24GB VRAM
#SBATCH --mem=96G                        # memory (optional override)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/robust_training/outs/advtrain/main/convnextsmall/%A/%a.out
#SBATCH --array=1-200%8


cd /home/ashtomer/projects/ares/robust_training/sh_files || exit 1

bash /home/ashtomer/projects/ares/robust_training/sh_files/train_models.sh