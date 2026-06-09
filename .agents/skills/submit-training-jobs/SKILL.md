---
name: submit-training-jobs
description: "Translate natural-language `ares` model-training requests into explicit approval-gated Slurm submissions using the repo's standard or epsilon-curriculum launchers, then verify only the newly submitted job's Slurm state and log after 30 seconds. Use when the user asks to launch, submit, train, continue, resume, or move a model from one epsilon to another on `rtx6000` or `rtx_pro_6000`."
---

# Goal

Make `ares` training submissions quick and predictable. Infer the intended launcher and job name from the user's text, show the interpretation, request approval for the exact outside-sandbox submission command before running `sbatch`, and inspect only the new job's Slurm state and log after approval.

Do not scan existing jobs, inspect unrelated logs, or rediscover naming rules from many files. Use this skill as the primary reference. Read a listed repo file only when the request uses an unsupported or unclear variant.

# Defaults

- Use ConvNeXt Small when the backbone is omitted.
- Use Madry when the criterion is omitted.
- Use pixel-space training unless the user says V1.
- Use `rtx_pro_6000` when the partition is omitted.
- Always submit the sequential resume array `1-10%1`.
- Ask one short question when an epsilon continuation omits `direct` versus `ramp`; never infer that scientific choice.
- Treat `RTX Pro`, `RTX Pro 6000`, and `rtx_pro_6000` as `rtx_pro_6000`.
- Treat `RTX6000`, `RTX 6000`, and `rtx6000` as `rtx6000`.

# Intent Mapping

## Standard Training

Use `sbatches/golan-trainmodels.sbatch`.

Build the job name from:

```text
[BACKBONE_PREFIX][MODE]_init<INIT>
```

Backbone prefixes:

- Small/default: no prefix
- Base: `convnext_base_`
- Large: `convnext_large_`

Modes:

- Clean baseline: `baseline`
- Madry: `linf_<EPS>`, `l2_<EPS>`, or `l1_<EPS>`
- TRADES: `linftrades_<EPS>`, `l2trades_<EPS>`, or `l1trades_<EPS>`
- GradNorm: `gradnorm_l1_<EPS>` or `gradnorm_l2_<EPS>`; the norm after `gradnorm_` is the penalty norm
- V1 clean: `v1clean`
- V1 neuronal noise: `v1noise`
- V1 Madry: `v1clean_linf_<EPS>`, `v1clean_l2_<EPS>`, or `v1clean_l1_<EPS>`
- V1 TRADES: `v1clean_linftrades_<EPS>`, `v1clean_l2trades_<EPS>`, or `v1clean_l1trades_<EPS>`

Unsupported combinations: V1 GradNorm, V1-noise adversarial training, and GradNorm without an explicit L1/L2 penalty norm.

Examples:

- “train linf madry eps 8 init 1” -> `linf_8_init1`
- “train base l2 trades eps 4 init 2 on rtx6000” -> `convnext_base_l2trades_4_init2`
- “train v1 clean linf madry eps 8 init 1” -> `v1clean_linf_8_init1`

## Epsilon Continuation

Use `sbatches/epsilon_curriculum.sbatch`.

Build the job name from:

```text
[BACKBONE_PREFIX][MODE]_cont<SOURCE_EPS>to<TARGET_EPS>_<direct|ramp>_init<INIT>
```

Supported modes are pixel or V1-clean Madry/TRADES modes from standard training. GradNorm and V1-noise continuation are unsupported.

- `direct`: fixed target epsilon for 30 epochs.
- `ramp`: warmup/ramp/fixed schedule for 40 epochs.
- The launcher starts from the source model's `model_best_adv.pth.tar`, falling back to `model_best.pth.tar`.
- Later array tasks resume the continuation model from `last.pth.tar`.

Example:

- “train linf madry from eps 4 to 8 on init 1” -> ask `direct` or `ramp`, then use `linf_cont4to8_<protocol>_init1`.

# Required Workflow

1. Parse the request into launcher, backbone, mode, source/target epsilon when relevant, init, protocol, partition, and array.
2. Ask only for a missing value that materially changes the job. In particular, ask for `direct` versus `ramp` when omitted.
3. Before submission, state the interpretation and exact job name in one compact summary.
4. Before running any command that contains `sbatch`, request user approval for the exact command that will submit and verify the job. Use `functions.exec_command` with `sandbox_permissions="require_escalated"` and a `justification` that asks whether to submit this specific training job. If approval is denied or unavailable, do not submit the job.
5. Run the approved command from the repository root. Resolve it from the current workspace instead of assuming a fixed absolute path. Use `sbatch --parsable`, capture the returned array job ID, wait exactly 30 seconds, then inspect only that job's Slurm state and task 1 log.
6. Report the submitted job ID and one of:
   - `launch confirmed`: the log identifies the expected job/model and reaches runtime initialization or training startup without an early error.
   - `submitted, not started after 30 seconds`: the new log does not exist yet.
   - `launch failed`: the new log contains an early error, traceback, rejected job name, missing continuation checkpoint, or mismatched parsed configuration.

Use this command shape, substituting the resolved values:

```bash
/usr/bin/bash -lc 'repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
job_id=$(sbatch --parsable --job-name=<JOB_NAME> --partition=<PARTITION> --array=1-10%1 <SBATCH_PATH>)
job_id=${job_id%%;*}
echo "SUBMITTED_JOB_ID=$job_id"
sleep 30
echo "NEW_JOB_SLURM_STATE"
squeue -j "$job_id" -o "%.18i %.16P %.45j %.2t %.10M %.40R" || true
sacct -j "$job_id" --format=JobID,JobName%45,Partition,State,Elapsed,ExitCode -n -X || true
log="<LOG_ROOT>/$job_id/1.out"
echo "NEW_JOB_LOG=$log"
if [[ -f "$log" ]]; then
  tail -n 160 "$log"
else
  echo "NEW_JOB_LOG_NOT_CREATED_AFTER_30_SECONDS"
fi'
```

Use `squeue` and `sacct` only with the newly returned job ID. Do not use broad Slurm queries, `scontrol`, broad `find`, or unrelated log reads as part of this workflow.

# Verification

Expected log roots:

- Standard: `outs/advtrain/golan_neuro/convnextsmall/<JOB_ID>/1.out`
- Continuation: `outs/advtrain/epsilon_curriculum/<JOB_ID>/1.out`

Confirm that the new log contains the expected `JOB NAME`, parsed fields, and `model_name`. Strong successful-start signals include `Runtime distributed=`, `Experiment:`, `Creating model:`, or `Start training for`.

Treat `[ERROR]`, `Traceback`, parser rejection, a missing source checkpoint, a different job/model name, or a failed Slurm state as failure. A pending/running job with no log after 30 seconds is not a failure.

# Relevant Relative Paths

- Standard sbatch: `sbatches/golan-trainmodels.sbatch`
- Continuation sbatch: `sbatches/epsilon_curriculum.sbatch`
- Standard naming reference: `sbatches/golan-trainmodels_jobname_patterns.txt`
- Standard parser/resume rules: `sbatches/train_launcher_lib.sh`
- Continuation parser/resume rules: `sbatches/cont_train_launcher.sh`
- Standard logs: `outs/advtrain/golan_neuro/convnextsmall/`
- Continuation logs: `outs/advtrain/epsilon_curriculum/`
