#!/bin/bash
# Quick torch.compile debug on B200.
# Runs debug_compile.py inside the NGC PyTorch 25.10 container (CUDA 13.0,
# pre-tuned for B200) to check whether the crash is a CUDA-12.8/Triton bug.
#
# Submit from Botero after git push + git pull on AIRCC:
#
#   ssh aircc "cd /shared/cycle2_bgu_golan_prj/ashtomer/ares && sbatch aircc/debug_compile.sbatch"
#
# Output: /shared/cycle2_bgu_golan_prj/logs/debug-compile-<JOBID>.out
#
# If PASS on inductor → switch Dockerfile to FROM nvcr.io/nvidia/pytorch:25.10-py3
#                        and rebuild the ares-train image.
# If FAIL on inductor but PASS on cudagraphs/aot_eager → B200 Triton bug, use
#   model.compile_model=False for AIRCC until a fixed PyTorch release.

#SBATCH --job-name=debug-compile
#SBATCH --partition=sandbox
#SBATCH --qos=sandbox_owner_780
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/cycle2_bgu_golan_prj/logs/debug-compile-%j.out
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle2_bgu_golan_prj:/shared/cycle2_bgu_golan_prj

set -euo pipefail

REPO="/shared/cycle2_bgu_golan_prj/ashtomer/ares"
cd "${REPO}"

echo "====== JOB ${SLURM_JOB_ID} on $(hostname) ======"
python aircc/debug_compile.py
