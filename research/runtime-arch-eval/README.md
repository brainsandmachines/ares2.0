# runtime-arch-eval

Per-protocol / per-architecture training-runtime profiling.

`runtime_arch_eval.csv` collects one row per (protocol × ConvNeXt arch) combination:

| column | meaning |
|---|---|
| `protocol` | `madry_{linf,l2,l1}`, `trades_{linf,l2,l1}`, `gradnorm_{l1,l2}`, `v1madryl2`, `v1tradesl2`, `baseline` |
| `arch` | `small` / `base` / `large` |
| `bsz` | largest per-GPU batch size that passed the per-task training smoke search |
| `optimizer` | optimizer name (`cfg.optimizer.opt`) |
| `compile` | whether `torch.compile` was active (production policy) |
| `fullepochruntime` | `train_runtime + eval_runtime` (seconds) |
| `train_runtime` | estimated seconds for one full train epoch |
| `eval_runtime` | estimated seconds for the end-of-epoch regular (clean) eval |

Each array task first finds the largest valid batch size for its own
`protocol × arch` combo. The search tries `128, 256, 384, ... 1024` and stops at
the first OOM, using the last successful batch size for the final timing run.
Each candidate is tested with the real training entrypoint and the same
protocol/model/compile settings as the final run, but with a one-batch
runtime-probe window so the search stays short. Smoke logs and the search table
are written under:

```text
outs/advtrain/runtime_arch_eval/<array_job>/<task>/batch_size_smoke/
```

**The final runtimes are full-epoch *estimates*.** After batch-size selection,
the task times a steady-state window of batches (the first
`runtime_probe_warmup_batches` are skipped to exclude compile/cuDNN warmup), then
extrapolates: `measured_seconds / measured_batches × total_batches_in_loader`.
This keeps each task cheap (~25 train + ~25 eval batches) while approximating
real per-epoch cost. Checkpoint saving and the final AutoAttack/PGD sweep are
disabled for these timing-only runs.

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
