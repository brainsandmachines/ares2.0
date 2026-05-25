---
name: training-runtime-optimizer
description: Use when exploring `ares` training-loop runtime ideas under strict protocol-preservation constraints. This skill benchmarks locally first on the fast GPU, records tested ideas to avoid repetition across scheduled calls, protects current attack/data semantics with focused tests, tries one additive runtime idea at a time, and emits Slurm micro-benchmarks only for protocol-specific ideas that already improved local runtime. Do not use for changing scientific protocol, attack math, augmentations, mixup semantics, or batch size as a final optimization.
---

# Goal

Explore simple runtime optimization ideas without changing the scientific meaning of the experiment.

This skill is for `ares` only. It must preserve the current training contract unless the user explicitly asks for a protocol change.

The skill should know where to search for ideas, which repo files define the training protocol, and which runtime-only settings can be changed safely to reduce runtime.

The skill is designed for repeated calls, including scheduled runs every 30 or 60 minutes. Each call should test one new idea, record the result, and avoid retesting ideas already present in `state/ideas_tested.json`.

When an idea wins for one protocol, do not stop there if the idea is related to other in-scope protocols. Promote the same idea into a bounded related-protocol sweep, testing the candidate against a fresh baseline for each related protocol before recommending code changes broadly.

# Protected Contract

Never silently change:

- `attack_it` or `v1_attack_it`
- `Linf`, `L2`, or `L1(l2_norm only)` step semantics
- `Linf`, `L2`, or `L1` projection semantics
- augmentation settings
- mixup settings
- dataset split or data roots used for the benchmark
- optimizer family
- learning-rate schedule family
- output/checkpoint naming semantics
- seed and reproducibility controls
- batch size as a final optimization

`l1_apgd` is out of scope for this skill. Ignore it unless the user explicitly asks for it.

Runtime-only configuration changes are allowed when they preserve data semantics. Examples: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, and equivalent DataLoader/runtime knobs. `batch_size` may only be used in the benchmark fallback ladder to avoid OOM; never recommend it as the final speedup.

# Required Workflow

Stage 1: idea and local Botero check.

1. Inspect the current training contract in `references/repo_contract.md`.
2. Inspect `state/ideas_tested.json` with `scripts/idea_state.py` and avoid repeating tested ideas for the same protocol and runtime knobs.
3. Look for one useful untested idea from the repo, existing outputs, profiler/timing evidence, or nearby code paths.
4. If repo-local exploration does not produce a useful untested idea, or the idea was already tested, search the web for current PyTorch/CUDA/DataLoader/runtime approaches. Record useful source URLs or notes with the idea.
5. Run a short test-coverage audit: name the protected behavior touched by the idea and the focused tests that protect it. If coverage is incomplete, require benchmark sanity checks for finite loss, unchanged protocol fields, unchanged data roots, and no protected config drift.
6. Establish a fresh local baseline on the fast GPU with `scripts/benchmark_local.sh`.
7. Run the focused contract suite with `scripts/run_contract_tests.sh`.
8. Try exactly one additive runtime change.
9. Re-run the contract tests before the candidate benchmark.
10. Run a bounded local benchmark on `imagenet_sample` when available.
11. Record throughput, iteration time, max VRAM, finite-loss sanity, source, pass/fail, and rejection reason in `state/ideas_tested.json`.
12. Compare the idea against the fresh local baseline for the same protocol.

Stage 2: scheduled exploration.

13. For repeated Botero/crontab calls, keep compact summaries and avoid retesting recorded ideas.
14. Keep full logs for winners, failures, or user-requested audits; otherwise prefer `result.json`, `summary.csv`, and `decision.json`.
15. Stop or narrow repeated jobs when memory pressure or excessive output files make the result harder to inspect.

Stage 3: Slurm verification.

16. If the candidate improves locally, identify all related in-scope protocols and run the same bounded baseline-vs-candidate check for each related protocol.
17. Record each related-protocol result separately in `state/ideas_tested.json`; a candidate can be a winner for one protocol and neutral or rejected for another.
18. Generate Slurm sbatches only for protocol-specific ideas that reduced local runtime.
19. Emit additive sbatches under `sbatches_botero/timing_skill/`.
20. After Slurm comparison, produce a clear recommendation: `recommend_code_change`, `run_full_epoch_validation`, or `reject_candidate`.

Stage 4: full-epoch validation.

21. Use full-epoch validation only when short Slurm evidence is not enough, such as `torch.compile` candidates or unclear stage-3 results.
22. For compile ideas, require evidence that compile was requested and applied, not only that runtime changed.
23. Do not edit the main training files to test candidates. Candidate changes must stay additive until Slurm or full-epoch evidence proves a winner.

# Protocols In Scope

- pixel-space ConvNeXt Small LINF L2 L1 Madry
- pixel-space ConvNeXt Small LINF L2 L1 TRADES, with and without radom start for the adversarial inner loop
- ConvNeXt Small GradNorm with `gradnorm_penalty_norm=l2`
- ConvNeXt Small V1 with `attack_domain=v1_feature`, `attack_norm=l2`, `attack_criterion=madry`
- ConvNeXt Small V1 with `attack_domain=v1_feature`, `attack_norm=l2`, `attack_criterion=trades`

# Benchmark Rules

- Use a bounded benchmark, not a full training job.
- Default to `1` warmup iteration and `5` measured iterations for local checks.
- Default worker checks to `num_workers in {6, 8}`.
- Default batch-size inspection to the fallback ladder `128 -> 112 -> 96`.
- Start at `128`. If that run fails with an OOM error, retry at `112`. If `112` also OOMs, retry at `96`.
- Do not run a broad batch-size sweep by default unless the user explicitly asks for a wider ceiling search.
- Per-protocol inspection should stay short. Five measured batches are the default signal for forward/backward timing checks.
- Slurm verification is different from the local short check. Generated sbatches should verify the winning idea with `20` measured training iterations on the target Slurm GPU.
- Reject any candidate that fails the contract suite, produces non-finite loss, or regresses local baseline timing.
- Do not emit an sbatch when an idea fails to improve the local runtime for that protocol.
- Save machine-readable outputs for both local and Slurm runs.
- Timing benchmarks must disable or stub real checkpoint saving, validation, final eval, and external logging unless the user explicitly asks to inspect those paths.
- Result files should be compact by default. Keep full logs only for winners, failures, or user-requested audits.

# Related-Protocol Sweep Rules

- A protocol is related when the candidate targets shared runtime mechanics used by that protocol without changing protected scientific behavior.
- Global runtime candidates such as `channels_last`, `dataloader_tuned`, `torch_compile`, and `zero_grad(set_to_none=True)` are related to all in-scope ConvNeXt Small protocols unless local evidence shows a protocol-specific incompatibility.
- DataLoader/runtime-only ideas are related across pixel, GradNorm, and V1 protocols because they preserve samples, augmentation, mixup, attack math, optimizer, and schedule.
- Pixel attack-loop ideas are related only to pixel-space protocols whose norm/criterion path uses the touched code. Do not automatically apply a Linf-specific kernel/allocation idea to L2, L1, V1, or GradNorm unless code inspection shows the same path is shared.
- TRADES-specific ideas are related only to TRADES protocols with the same random-start semantics unless the idea is clearly outside the random-start path.
- V1 feature-attack ideas are related only to V1 feature-attack protocols unless the touched code is shared with pixel-space attacks.
- GradNorm ideas are related only to GradNorm protocols unless the touched code is outside GradNorm regularization.
- If relation is ambiguous, test only the original protocol and record why the sweep was not expanded.
- Use `scripts/run_one_idea_cycle.py --related-protocols auto` for built-in global runtime candidates, or pass an explicit comma-separated protocol list for custom related sweeps.

# Idea Exploration Rules

- Do not restrict exploration to a fixed list of candidates. Existing ideas like DataLoader tuning, `channels_last`, `torch.compile`, and `zero_grad(set_to_none=True)` are examples, not a closed set.
- Before benchmarking, define an `idea_id` that is stable enough to detect repeats, such as `pixel_linf_madry__fused_attack_noise_init`.
- The idea must be concrete enough to test in the current call.
- Candidate experiments may use wrappers, monkeypatches, temporary runner changes, CLI overrides, or a narrow benchmark harness change.
- If an idea needs a new additive harness path, add it in the skill scripts or a temporary candidate file, not in the main training implementation.
- Record all outcomes, including failed and neutral ideas, so scheduled calls continue exploring.

# Final Decision Format

End with one of these decisions:

- `this suggested change achieved faster runtime; do the following changes to your code: <implementation plan>`
- `this suggested change needs full-epoch validation before a code change; run: <validation plan>`
- `this change did not improve runtime or failed validation; no code change is recommended`

Include protocol, idea id, baseline throughput, candidate throughput, speedup ratio, test status, artifact paths, and generated sbatches when present.

If a related-protocol sweep ran, summarize winners and non-winners separately. Recommend a broad code/config change only for protocols where the candidate improved locally; for mixed results, recommend protocol-specific enablement.

Every `decision.json` or Slurm comparison summary should include a machine-readable recommendation with `stage`, `recommendation`, `reason`, `required_next_validation`, `proposed_code_change`, and `evidence_paths`.

# Files To Read

- `references/repo_contract.md`
- `references/optimization_rules.md`
- `references/helpful_docs.md`
- `state/ideas_tested.json` when present

# Scripts

- `scripts/run_contract_tests.sh`
- `scripts/benchmark_local.sh`
- `scripts/run_candidate_benchmark.py`
- `scripts/make_slurm_benchmarks.py`
- `scripts/slurm_baselines.py`
- `scripts/compare_benchmarks.py`
- `scripts/idea_state.py`
- `scripts/run_one_idea_cycle.py`
