#!/bin/bash
# Daily AutoAttack-grid watcher, run from a Botero cron. Over ssh, on the Slurm login node,
# runs `python -m slurm_job_manager.reopen_eval --apply --all`: finds finished, fully-trained
# sjm rows whose linf/l2/l1 x eps grid is incomplete on best/last/advbest (trained under the
# old eps_norm default, or final_eval raised and the row was marked finished anyway) and puts
# them back to pending for an eval-only rerun. Skips aa_sweep excludes and models the aa_sweep
# queue still holds; a model re-opened twice without completing is reported as capped instead.
# Emails when it re-opens or caps something, or fails.
#
# Install (Botero crontab -e) -- 07:00, so its mail lands in the 07:30 digest:
#   0 7 * * * /home/tomer_a/Documents/ares/slurm_job_manager/scripts/reopen_eval_daily.sh >> /home/tomer_a/Documents/ares/slurm_job_manager/logs/reopen_eval.log 2>&1
#
#   slurm_job_manager/scripts/reopen_eval_daily.sh --dry-run   # list only, change nothing

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
SSH_HOST="${SJM_SSH_HOST:-slurm}"
CLUSTER_REPO="${SJM_CLUSTER_REPO:-/home/ashtomer/projects/ares}"
# reopen_eval imports torch/omegaconf (via the eval helpers): the login node's python3 has neither.
CLUSTER_PYTHON="${SJM_CLUSTER_PYTHON:-/home/ashtomer/.conda/envs/tomer_advtrain/bin/python}"
LOCK_FILE="${SJM_REOPEN_EVAL_LOCK:-$REPO_ROOT/slurm_job_manager/logs/.reopen_eval.lock}"

mkdir -p -m 0755 "$(dirname "$LOCK_FILE")"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[reopen_eval] $(date -Is) SKIP: previous run still holds $LOCK_FILE" >&2
    exit 0
fi

dry_run=0
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && dry_run=1
done
apply_args="--apply --all"
[[ "$dry_run" -eq 1 ]] && apply_args=""

notify() {
    local subject="$1" body="$2" dedup="${3:-}"
    [[ "$dry_run" -eq 1 ]] && return 0
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python3 - "$subject" "$body" "$dedup" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body, dedup = sys.argv[1:4]
emailer = make_emailer(source="sjm.reopen_eval")
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body, dedup_key=dedup or None)
PYEOF
}

remote_cmd="cd $CLUSTER_REPO && PYTHONPATH=$CLUSTER_REPO $CLUSTER_PYTHON -W ignore -m slurm_job_manager.reopen_eval --db $CLUSTER_REPO/slurm_job_manager/jobs.sqlite $apply_args"

run_remote() {
    ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 "$SSH_HOST" "$remote_cmd" 2>&1
}

echo "[reopen_eval] $(date -Is) start${apply_args:+ ($apply_args)}"
out="$(run_remote)"
rc=$?
# ssh itself failed (rc 255); `ssh slurm` round-robins login nodes, so one retry usually lands on
# a healthy one. A rerun is safe: reopen_for_eval only touches rows that are still finished.
if [[ "$rc" -eq 255 ]]; then
    echo "[reopen_eval] $(date -Is) ssh rc=255, retrying once" >&2
    sleep 15
    out="$(run_remote)"
    rc=$?
fi
[[ -n "$out" ]] && printf '%s\n' "$out"

summary="$(grep -E '^\[reopen_eval\]( DRY-RUN)? summary ' <<<"$out" | tail -1)"
if [[ "$rc" -ne 0 || -z "$summary" ]]; then
    echo "[reopen_eval] $(date -Is) ERROR: remote run failed rc=$rc" >&2
    notify "[sjm] reopen_eval failed (rc=$rc)" "$out" "sjm-reopen-eval-failed"
    exit $(( rc == 0 ? 1 : rc ))
fi

reopened_n="$(grep -oP 'reopened=\K[0-9]+' <<<"$summary")"
capped_n="$(grep -oP 'capped=\K[0-9]+' <<<"$summary")"

if [[ "$capped_n" -gt 0 ]]; then
    notify "[sjm] reopen_eval: $capped_n model(s) still missing AA cells after repeated re-opens" "$out" \
        "sjm-reopen-eval-capped"
fi
if [[ "$reopened_n" -gt 0 ]]; then
    notify "[sjm] reopen_eval re-opened $reopened_n model(s) for an eval-only AA rerun" "$out"
fi
echo "[reopen_eval] $(date -Is) done"
exit 0
