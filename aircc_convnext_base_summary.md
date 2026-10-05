# AIRCC usage summary: ConvNeXt-Base (ImageNet)

*Status as of 2026-09-28. Scope: ConvNeXt-Base runs only.*

## What ran
- All planned ConvNeXt-Base conditions ran on the AIRCC B200 allocation except V1 TRADES.
- Clean, V1 clean and V1 noise baselines trained for 150 epochs. So did adversarial training (Madry, TRADES, V1 Madry).
- Epsilon levels 1, 2 and 4 were trained from scratch. Levels 6 and 8 were reached by continuing from the ε=4 checkpoint for 40 more epochs at a lower LR (1e-4).
- The AIRCC allocation ended with 116 grid models finished. We finished the remaining runs on the lab GPUs (BGU Slurm cluster): 19 more models.

## Changes from the original plan
- **Random initializations went from 3 to 2 per protocol.** Many runs failed and had to be restarted: catastrophic overfitting, gradient collapse, and training collapse at large ε. The restarts used up allocation time, so the 3rd initialization (init2) was never launched.
- **DVD conditions were added.** "Developmental visual diet" is age-curve data augmentation. The new conditions are DVD clean, DVD Madry and DVD TRADES, each over 3 norms × 5 ε, trained for 200 epochs.
- **More PGD steps for harder runs:**
  - From-scratch runs: 3 PGD steps (unchanged).
  - Continuation runs (ε 4→6, 4→8): 5 steps, or 7 for L2 4→8.
  - DVD ε=4 runs: raised from 3 to 5 steps after repeated failures at 3.
- **Input-gradient regularization (L1) collapsed at large ε.** The ε = 4, 6 and 8 runs failed repeatedly, so only ε = 1 and 2 finished. Two more are still training on the lab GPUs.
- **V1:** V1 Madry ran with L2 only, as planned.

## Planned vs. completed

| Condition | Planned (3 inits) | Finished | Notes |
|---|---|---|---|
| Clean | 3 | 2 | |
| Madry (L1/L2/L∞ × 5 ε) | 45 | 30 | full grid, 2 inits |
| TRADES (L1/L2/L∞ × 5 ε) | 45 | 30 | full grid, 2 inits |
| Input-gradient regularization | 15 | 3 (+2 running) | large ε collapsed |
| V1 clean | 3 | 2 | |
| V1 random noise | 3 | 2 | |
| V1 Madry (L2 × 5 ε) | 15 | 8 (+1 running) | |
| V1 TRADES | 15 | 0 | not run |
| **Original grid total** | **144** | **77** | |
| DVD clean *(new)* | – | 2 | |
| DVD Madry *(new)* | – | 30 | full grid, 2 inits |
| DVD TRADES *(new)* | – | 26 (+2 running) | |
| **Total incl. DVD** | | **135 (+5 running)** | |

- A few extra runs are not counted above:
  - 5 short pilot runs, used to choose the continuation PGD steps and LR.
  - 6 DVD "reset-epoch" continuation variants. They scored lower than the standard continuation and were discontinued.
