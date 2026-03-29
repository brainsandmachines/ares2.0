Final Eval (AutoAttack + PGD Sweep)

This README gives a copy‑paste SLURM `sbatch` script to run final evaluation on an already trained model checkpoint.
It runs AutoAttack and PGD sweep, and saves CSVs + plots in the same folder as the checkpoint.

Quick Start (SBATCH)
```bash
#!/bin/bash
#SBATCH --job-name=final_eval
#SBATCH --output=logs/final_eval_%j.out
#SBATCH --error=logs/final_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

module load anaconda
source activate tomer_advtrain_pro

cd /home/ashtomer/projects/ares

CKPT="/home/ashtomer/projects/ares/results/models/convnext_small_l2_16_init1/model_best.pth.tar"
VAL_DIR="/mnt/data/datasets/imagenet/val"
OUT_DIR="$(dirname "${CKPT}")"

python data_analysis/final_eval.py \
  --checkpoint "${CKPT}" \
  --val-dir "${VAL_DIR}" \
  --aa --pgd --plots \
  --device cuda \
  --out-dir "${OUT_DIR}"
```

Optional Overrides
- Limit to 1 batch for fast sanity check:
```
--aa-max-batches 1 --pgd-max-batches 1
```
- Force AutoAttack norm/eps if parsing fails:
```
--aa-norm L2 --aa-eps 16
```
- Customize PGD eps or norms:
```
--pgd-eps 0.5,1,2,4,8,16 --pgd-norms linf,l2,l1
```

Outputs
- AutoAttack CSV: `<checkpoint_dir>/autoattack_results.csv`
- PGD CSV: `<checkpoint_dir>/pgd_validation_results.csv`
- Plots: `<checkpoint_dir>/plots/...`
