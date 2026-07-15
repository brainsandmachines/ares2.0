#!/bin/bash
# Daily sjm (Botero Slurm campaign) check-in: runs the slurm-status skill's
# status script and emails if any running job is flagged with a stale
# heartbeat (hb > 900s -- informational only, sjm has no heartbeat-based
# requeue) or low CPU utilization (AveCPU/Elapsed < 50% via a live sstat
# query -- the reliable stuck-job signal; heartbeat_ts alone can look fresh
# during a real multi-day hang, see slurm_status.py's module docstring).
# Run from a Botero cron.
#
# Install (Botero crontab -e):
#   0 6 * * * /home/tomer_a/Documents/ares/slurm_job_manager/scripts/check_slurm_status.sh >> /home/tomer_a/Documents/ares/slurm_job_manager/logs/status_check.log 2>&1

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
STATUS_SCRIPT="${SJM_STATUS_SCRIPT:-$HOME/.claude/skills/slurm-status/scripts/slurm_status.py}"
LOCK_FILE="${SJM_STATUS_LOCK:-$REPO_ROOT/slurm_job_manager/logs/.status_check.lock}"

mkdir -p -m 0755 "$(dirname "$LOCK_FILE")"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[status] $(date -Is) SKIP: another status check already holds $LOCK_FILE" >&2
    exit 0
fi

notify() {
    local subject="$1" body="$2"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python3 - "$subject" "$body" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body = sys.argv[1], sys.argv[2]
emailer = make_emailer()
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body)
PYEOF
}

cd "$REPO_ROOT" || exit 1

status_out="$(python3 "$STATUS_SCRIPT" 2>&1)"
status_rc=$?

if [[ "$status_rc" -ne 0 ]]; then
    echo "[status] $(date -Is) ERROR: slurm_status.py failed rc=$status_rc" >&2
    notify "[sjm] status check failed" "$STATUS_SCRIPT exited rc=$status_rc:

$status_out"
    exit "$status_rc"
fi

old_hb_n="$(grep -oP 'old-hb>\d+s: \K[0-9]+' <<<"$status_out" | head -1)"
stuck_n="$(grep -oP 'cpu-utilization check \([^)]+\): \K[0-9]+(?= possibly stuck)' <<<"$status_out" | head -1)"
cpu_unavailable=0
grep -q '^## cpu-utilization check: unavailable' <<<"$status_out" && cpu_unavailable=1

old_hb_n="${old_hb_n:-0}"
stuck_n="${stuck_n:-0}"

echo "[status] $(date -Is) old_hb=$old_hb_n cpu_stuck=$stuck_n cpu_unavailable=$cpu_unavailable"

if [[ "$old_hb_n" -gt 0 || "$stuck_n" -gt 0 ]]; then
    echo "[status] $(date -Is) ALERT: old_hb=$old_hb_n cpu_stuck=$stuck_n" >&2
    notify "[sjm] status alert: old_hb=$old_hb_n cpu_stuck=$stuck_n" "$status_out"
elif [[ "$cpu_unavailable" -eq 1 ]]; then
    echo "[status] $(date -Is) note: cpu-utilization check unavailable (ssh unreachable this run), heartbeats clean"
else
    echo "[status] $(date -Is) ok: no stale/stuck jobs"
fi

exit 0
