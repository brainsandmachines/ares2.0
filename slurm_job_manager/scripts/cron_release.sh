#!/bin/bash
# Hourly sjm owner sweep, run from a Botero cron. Over ssh, on the Slurm login node,
# runs `python -m slurm_job_manager.cron_release` (JobDB.cron_release): releases DB
# rows whose owning task is not training (gone, CANCELLED, COMPLETING, PENDING after a
# NODE_FAIL requeue, or restarted under the same id) and scancels owners whose
# heartbeat stalled. Emails when it scancels, trips the many-stalled breaker, or fails.
#
# Install (Botero crontab -e):
#   15 * * * * /home/tomer_a/Documents/ares/slurm_job_manager/scripts/cron_release.sh >> /home/tomer_a/Documents/ares/slurm_job_manager/logs/cron_release.log 2>&1
#
#   slurm_job_manager/scripts/cron_release.sh --dry-run   # report only, change nothing

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
SSH_HOST="${SJM_SSH_HOST:-slurm}"
CLUSTER_REPO="${SJM_CLUSTER_REPO:-/home/ashtomer/projects/ares}"
LOCK_FILE="${SJM_CRON_RELEASE_LOCK:-$REPO_ROOT/slurm_job_manager/logs/.cron_release.lock}"

mkdir -p -m 0755 "$(dirname "$LOCK_FILE")"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[cron_release] $(date -Is) SKIP: previous sweep still holds $LOCK_FILE" >&2
    exit 0
fi

dry_run=0
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && dry_run=1
done

notify() {
    local subject="$1" body="$2" dedup="${3:-}"
    [[ "$dry_run" -eq 1 ]] && return 0
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python3 - "$subject" "$body" "$dedup" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body, dedup = sys.argv[1:4]
emailer = make_emailer(source="sjm.cron_release")
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body, dedup_key=dedup or None)
PYEOF
}

remote_args=""
# Guarded: with no arguments, `printf ' %q'` still emits one empty '' argument.
[[ $# -gt 0 ]] && remote_args="$(printf ' %q' "$@")"
remote_cmd="cd $CLUSTER_REPO && SJM_DB=$CLUSTER_REPO/slurm_job_manager/jobs.sqlite PYTHONPATH=$CLUSTER_REPO python3 -m slurm_job_manager.cron_release$remote_args"

run_remote() {
    ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 "$SSH_HOST" "$remote_cmd" 2>&1
}

echo "[cron_release] $(date -Is) start"
out="$(run_remote)"
rc=$?
# ssh itself failed (rc 255); `ssh slurm` round-robins login nodes, so one retry
# usually lands on a healthy one. A rerun is safe: every release is claim-guarded.
if [[ "$rc" -eq 255 ]]; then
    echo "[cron_release] $(date -Is) ssh rc=255, retrying once" >&2
    sleep 15
    out="$(run_remote)"
    rc=$?
fi
[[ -n "$out" ]] && printf '%s\n' "$out"

summary="$(grep -E '^\[cron_release\]( DRY-RUN)? summary ' <<<"$out" | tail -1)"
if [[ "$rc" -ne 0 || -z "$summary" ]]; then
    echo "[cron_release] $(date -Is) ERROR: remote sweep failed rc=$rc" >&2
    notify "[sjm] cron_release failed (rc=$rc)" "$out" "sjm-cron-release-failed"
    exit $(( rc == 0 ? 1 : rc ))
fi

cancelled_n="$(grep -oP 'cancelled=\K[0-9]+' <<<"$summary")"
breaker="$(grep -oP 'breaker=\K[0-9]+' <<<"$summary")"

if [[ "$breaker" -eq 1 ]]; then
    notify "[sjm] cron_release breaker tripped: stalled rows NOT cancelled" "$out" \
        "sjm-cron-release-breaker"
elif [[ "$cancelled_n" -gt 0 ]]; then
    notify "[sjm] cron_release scancelled $cancelled_n stalled job(s)" "$out"
fi
echo "[cron_release] $(date -Is) done"
exit 0
