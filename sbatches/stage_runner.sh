#!/bin/bash
# In-job stage machine, sourced by the training launchers.
#
# A single job walks the deterministic pipeline for its model row:
#   TRAIN -> AA_SWEEP -> PLOT_RESULTS -> COMPLETED
# reading/advancing state in the orchestrator SQLite DB via `orchestrator.jobctl`.
# State is keyed by ORCH_MODEL_ID / ORCH_DB, exported by the orchestrator through
# `sbatch --export`. If those are unset (a manual run), training runs unchanged.
#
# Usage from a launcher (after it has built the base training command array):
#   source sbatches/stage_runner.sh
#   orch_pipeline "${cmd[@]}"
#
# The launcher must also have `model_name` and `MODELS_ROOT` available for the
# plotting step (derived from `model_dir` here as a fallback).

orch_pipeline() {
    local base_cmd=("$@")

    # ---- not orchestrated: behave exactly like before ----
    if ! python -m orchestrator.jobctl tracked 2>/dev/null; then
        echo "[stage_runner] not orchestrated (ORCH_MODEL_ID unset); running training directly"
        "${base_cmd[@]}"
        return $?
    fi

    local stage status model_dir total st
    stage="$(python -m orchestrator.jobctl get current_stage)"
    status="$(python -m orchestrator.jobctl get status)"
    model_dir="$(python -m orchestrator.jobctl get model_dir)"
    total="$(python -m orchestrator.jobctl get total_epochs)"
    echo "[stage_runner] model=$ORCH_MODEL_ID stage=$stage status=$status total_epochs=$total dir=$model_dir"

    # ---- fast inspection: nothing to do ----
    if [[ "$stage" == "COMPLETED" ]]; then
        echo "[stage_runner] already COMPLETED; zero-overhead exit 0"
        return 0
    fi

    # The DB row is authoritative for run length: pass training.epochs=<total>
    # unless the launcher already set it (e.g. curriculum). This also guarantees
    # the AA stage resumes AT max epoch (training loop skipped, AA only).
    local epochs_args=()
    if [[ "${base_cmd[*]}" != *"training.epochs="* ]]; then
        epochs_args=(training.epochs="$total")
    fi

    # ---- TRAIN (train only; AA is its own stage) ----
    if [[ "$stage" == "TRAIN" ]]; then
        echo "[stage_runner] === STAGE TRAIN ==="
        "${base_cmd[@]}" "${epochs_args[@]}" final_eval=False
        st=$?
        if [[ $st -ne 0 ]]; then echo "[stage_runner] TRAIN failed ($st)"; return $st; fi
        python -m orchestrator.jobctl advance AA_SWEEP
        stage="AA_SWEEP"
    fi

    # ---- AA_SWEEP (AutoAttack only on best,last,advbest; no PGD) ----
    if [[ "$stage" == "AA_SWEEP" ]]; then
        echo "[stage_runner] === STAGE AA_SWEEP ==="
        # base_cmd already resumes from last.pth.tar; at max epoch the training
        # loop is skipped and only the AutoAttack sweep runs (skip-if-complete
        # handles best/last/advbest re-runs).
        "${base_cmd[@]}" "${epochs_args[@]}" final_eval=True final_eval_autoattack=True final_eval_pgd=False
        st=$?
        if [[ $st -ne 0 ]]; then echo "[stage_runner] AA_SWEEP failed ($st)"; return $st; fi
        python -m orchestrator.jobctl advance PLOT_RESULTS
        stage="PLOT_RESULTS"
    fi

    # ---- PLOT_RESULTS (compare best/last/advbest, pick optimal checkpoint) ----
    if [[ "$stage" == "PLOT_RESULTS" ]]; then
        echo "[stage_runner] === STAGE PLOT_RESULTS ==="
        local mname mroot
        mname="$(basename "$model_dir")"
        mroot="$(dirname "$model_dir")"
        python data_analysis/plot_autoattack_comparation.py \
            --plot_name "$mname" \
            --models "$mname" \
            --checkpoint-kind all \
            --models-root "$mroot" \
            --out-dir "$model_dir" \
            || echo "[stage_runner] PLOT_RESULTS failed (non-fatal); continuing"
        python -m orchestrator.jobctl advance COMPLETED
        stage="COMPLETED"
    fi

    echo "[stage_runner] pipeline finished; stage=$stage"
    return 0
}
