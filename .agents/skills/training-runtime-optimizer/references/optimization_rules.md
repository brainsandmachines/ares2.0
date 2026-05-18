# Optimization Rules

## Local-First Flow

1. Load `state/ideas_tested.json` and skip ideas already tested for the same protocol and runtime-relevant knobs.
2. Inspect repo code, existing benchmark outputs, and timing/profiler evidence for one useful untested idea.
3. Search the web only when repo-local exploration finds no useful untested idea, or the repo idea has already been tested.
4. Run a focused test-coverage audit for the selected idea and protocol.
5. Measure a fresh local baseline.
6. Run the contract suite.
7. Apply one additive runtime candidate.
8. Re-run the contract suite.
9. Benchmark locally.
10. If the candidate wins locally, benchmark the same candidate against fresh baselines on all related in-scope protocols.
11. Record the primary and each related-protocol result in `state/ideas_tested.json` whether it wins, fails, or is neutral.
12. Emit Slurm sbatches only for protocol-specific local winners.

Default local check profile:

- `warmup_iters=1`
- `measured_iters=5`
- worker checks at `num_workers=6` and `num_workers=8`
- batch-size fallback ladder `128 -> 112 -> 96`
- only fall back to the next smaller batch size on OOM

## Candidate Rules

- Candidate experiments must stay additive.
- Do not modify the main training files while evaluating candidates.
- Candidate experiments may wrap or monkeypatch runtime behavior from a separate script.
- Candidates must preserve attack semantics, data semantics, and reproducibility settings.
- Exploration is open-ended. Do not constrain the skill to a fixed candidate enum or registry.
- Existing candidates such as DataLoader tuning, `channels_last`, `torch.compile`, and `zero_grad(set_to_none=True)` are examples only.
- Every idea must have a stable `idea_id`, source, touched behavior, and rejection/win status.
- When recording a related-protocol result, adapt protocol-prefixed idea IDs to the related protocol, e.g. `pixel_linf_madry__channels_last` becomes `pixel_l2_madry__channels_last`.

## Related Protocols

- Global runtime candidates (`channels_last`, DataLoader tuning, `torch.compile`, `zero_grad(set_to_none=True)`) should be tested across all in-scope protocols after a primary local win.
- Protocol-path candidates should be tested only on protocols that share the touched code path.
- If relation is unclear, keep the result scoped to the original protocol and record the reason.
- Recommend broad enablement only for protocols that individually beat their fresh baseline.

## Exploration Guidance

- Prefer repo-local ideas first: DataLoader bottlenecks, attack-loop host overhead, tensor layout conversions, redundant CUDA synchronization, optimizer zeroing, compile boundaries, GradNorm overhead, V1 feature-attack overhead, and repeated allocation.
- Use web search for current PyTorch/CUDA/DataLoader/runtime approaches only after local idea mining is exhausted or repetitive.
- Record useful source URLs or notes for web-derived ideas.
- Tune runtime-only config when safe: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, and equivalent DataLoader/runtime settings.
- Never recommend `batch_size` as the final speedup. The fallback ladder exists only to avoid OOM during timing probes.

## Reject Conditions

Reject the candidate immediately if:

- contract tests fail
- loss becomes non-finite
- benchmark crashes
- timing regresses versus baseline
- the candidate changes protected protocol knobs
- the candidate relies on `batch_size` as the final optimization
- the candidate cannot be distinguished from an idea already recorded in `ideas_tested.json`

## Slurm Emission Rule

Generate sbatches only for ideas that already won the fresh local comparison.

Slurm emission requirements:

- an sbatch is allowed only when a concrete idea improved local runtime for a specific protocol
- if the current idea iteration does not beat the local baseline, emit no sbatch
- generated sbatches should verify the idea with `20` measured training iterations on the target Slurm GPU
- generated sbatches remain additive and protocol-specific

Output path:

- `sbatches_botero/timing_skill/`

Those sbatches must be additive and must not replace existing training sbatches.
