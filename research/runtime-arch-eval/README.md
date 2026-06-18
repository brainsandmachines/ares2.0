# runtime-arch-eval

Per-protocol / per-architecture training-runtime profiling.

`runtime_arch_eval.csv` collects one row per (protocol × ConvNeXt arch) combination:

| column | meaning |
|---|---|
| `protocol` | `madry_{linf,l2,l1}`, `trades_{linf,l2,l1}`, `gradnorm_{l1,l2}`, `v1madryl2`, `v1tradesl2`, `baseline` |
| `arch` | `small` / `base` / `large` |
| `bsz` | per-GPU batch size used (chosen by GPU memory, via `parse_train_job`) |
| `optimizer` | optimizer name (`cfg.optimizer.opt`) |
| `compile` | whether `torch.compile` was active (production policy) |
| `fullepochruntime` | `train_runtime + eval_runtime` (seconds) |
| `train_runtime` | estimated seconds for one full train epoch |
| `eval_runtime` | estimated seconds for the end-of-epoch regular (clean) eval |

**The runtimes are full-epoch *estimates*.** Each run times a steady-state window of
batches (the first `runtime_probe_warmup_batches` are skipped to exclude
compile/cuDNN warmup), then extrapolates: `measured_seconds / measured_batches ×
total_batches_in_loader`. This keeps each task cheap (~25 train + ~25 eval batches)
while approximating real per-epoch cost. Checkpoint saving and the final
AutoAttack/PGD sweep are disabled for these timing-only runs.

## Run

```bash
# from the repo root, on the SLURM submit node
sbatch sbatches_botero/runtime_epoch_inspection.sbatch          # full 33-task array
sbatch --array=0 sbatches_botero/runtime_epoch_inspection.sbatch # single smoke-test task
```

## Inspect

```bash
squeue -u "$USER"
column -t -s, research/runtime-arch-eval/runtime_arch_eval.csv
```

The mechanism is the `runtime_probe*` keys in `robust_training/configs/config.yaml`
plus `ares/utils/runtime_probe.py` (`PhaseTimer`, `append_runtime_row`); the timers
are threaded into `train_one_epoch` / `validate`. With `runtime_probe=False`
(the default) normal training is unaffected.
