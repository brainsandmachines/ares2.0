#!/bin/bash
# Daily AIRCC check-in: (1) run the aircc-status skill's status script and email
# if the running count doesn't match squeue's live expected count (2 models per
# RUNNING aircc_jm task) or any job has failed, (2) back up the AIRCC
# results/models tree to Botero, pulled through the local sshfs AIRCC mount (no
# ssh in the rsync itself), emailing only if the rsync itself fails. Run from a
# Botero cron.
#
# Install (Botero crontab -e):
#   0 3 * * * /home/tomer_a/Documents/ares/aircc/aircc_job_manager/scripts/backup_aircc_models.sh >> /mnt/data/robustness_models/aircc_models/backup.log 2>&1
#
# Permissions: --no-perms/--no-owner/--no-group drop the sshfs-reported perms so
# files land owned by your Botero user; the dest root is created mode 0755.

set -u -o pipefail

# --- CONFIRM THESE PATHS ---
# Local sshfs mount of the AIRCC results/models dir, e.g.:
#   ~/aircc_mount/shared/cycle2_bgu_golan_prj/ashtomer/ares/results/models
SRC="${AIRCC_MOUNT:-$HOME/aircc_mount/ashtomer/ares/results/models}"
DEST="/mnt/data/robustness_models/aircc_models"
REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
STATUS_SCRIPT="${AIRCC_STATUS_SCRIPT:-$HOME/.claude/skills/aircc-status/scripts/aircc_status.py}"
LOCK_FILE="${AIRCC_BACKUP_LOCK:-$DEST/.backup.lock}"

mkdir -p -m 0755 "$DEST"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[backup] $(date -Is) SKIP: another backup run already holds $LOCK_FILE" >&2
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

# --- 1. aircc-status check ---
# expected_running comes from aircc_status.py itself, computed live from squeue
# (2 models per RUNNING aircc_jm array task) -- not a hardcoded slot count, since
# the number of live tasks (and therefore expected running models) changes as the
# campaign's job allocation changes.
status_out="$(python3 "$STATUS_SCRIPT" 2>&1)"
status_rc=$?
running_n="$(grep -oP 'counts:.*?\brunning=\K[0-9]+' <<<"$status_out" | head -1)"
expected_n="$(grep -oP '^expected_running=\K[0-9]+' <<<"$status_out" | head -1)"
check_state="$(grep -oP '\[(OK|MISMATCH)\]' <<<"$status_out" | head -1)"
failed_n="$(grep -oP '## failed \(\K[0-9]+' <<<"$status_out" | head -1)"

if [[ "$status_rc" -ne 0 ]]; then
    echo "[status] $(date -Is) ERROR: aircc_status.py failed rc=$status_rc" >&2
    notify "[aircc] status check failed" "$STATUS_SCRIPT exited rc=$status_rc:

$status_out"
elif [[ -z "$expected_n" ]]; then
    # squeue SSH check was unavailable -- can't compare against a live expected
    # count, so only alert on failures, don't fabricate a running-count mismatch.
    if [[ "${failed_n:-0}" -ne 0 ]]; then
        echo "[status] $(date -Is) ALERT: failed=${failed_n:-0} (expected_running check unavailable)" >&2
        notify "[aircc] status alert: failed=${failed_n:-0}" "$status_out"
    else
        echo "[status] $(date -Is) ok: running=${running_n:-0} failed=0 (expected_running check unavailable)"
    fi
elif [[ "$check_state" != "[OK]" || "${failed_n:-0}" -ne 0 ]]; then
    echo "[status] $(date -Is) ALERT: running=${running_n:-0} (expected $expected_n) failed=${failed_n:-0}" >&2
    notify "[aircc] status alert: running=${running_n:-0} (expected $expected_n) failed=${failed_n:-0}" "$status_out"
else
    echo "[status] $(date -Is) ok: running=$running_n (expected $expected_n) failed=0"
fi

# --- 2. rsync backup ---
if [[ ! -d "$SRC" ]]; then
    echo "[backup] ERROR: source not mounted/missing: $SRC" >&2
    echo "[backup] is the sshfs AIRCC mount up?" >&2
    notify "[aircc] backup source not mounted" "source not mounted/missing: $SRC
is the sshfs AIRCC mount up?"
    exit 1
fi

echo "[backup] $(date -Is) rsync $SRC/ -> $DEST/"
rsync_rc=0
# Capture rsync's stderr so we can distinguish benign "vanished" churn (the
# cluster deleting superseded checkpoints mid-transfer) from real read/IO errors.
# stdout (stats2/progress2) still flows straight to the cron log.
err_tmp="$(mktemp "${TMPDIR:-/tmp}/aircc_rsync_err.XXXXXX")"
# Skip the per-epoch intermediate checkpoints (checkpoint-N.pth.tar, ~1.4GB each);
# we only keep last.pth.tar, model_best*.pth.tar, configs, logs and summaries.
# --delete-excluded also purges any such checkpoints already on the destination.
# protect the script's own log/lock (they live in $DEST, not in $SRC, so the
# --delete flags would otherwise remove them mid-run).
rsync -rt --no-perms --no-owner --no-group --partial --delete-after --delete-excluded \
    --filter='protect /backup.log' --filter='protect /.backup.lock' \
    --exclude='checkpoint-*.pth.tar' --info=stats2,progress2 \
    "$SRC/" "$DEST/" 2>"$err_tmp" || rsync_rc=$?
cat "$err_tmp" >&2   # keep the stderr lines in the log

# Lines that are expected when backing up a live training tree. Anything on
# rsync's stderr that does NOT match one of these is treated as a real error.
benign_pat='^(file has vanished: .*|rsync warning: .*|rsync error: some files/attrs were not transferred.*|[[:space:]]*)$'

if [[ "$rsync_rc" -eq 0 ]]; then
    echo "[backup] $(date -Is) done"
elif [[ "$rsync_rc" -eq 23 || "$rsync_rc" -eq 24 ]]; then
    # rc=24: files vanished before transfer; rc=23: vanished mid-transfer / read
    # error. Forgive either ONLY if every stderr line is a known-benign vanish.
    real_errs="$(grep -vE "$benign_pat" "$err_tmp")"
    if [[ -n "$real_errs" ]]; then
        echo "[backup] $(date -Is) ERROR: rsync failed rc=$rsync_rc (non-vanished errors on stderr)" >&2
    else
        echo "[backup] $(date -Is) note: rc=$rsync_rc, only superseded checkpoints vanished mid-transfer"
        echo "[backup] $(date -Is) done"
        rsync_rc=0
    fi
else
    echo "[backup] $(date -Is) ERROR: rsync failed rc=$rsync_rc" >&2
fi

if [[ "$rsync_rc" -ne 0 ]]; then
    notify "[aircc] backup rsync failed rc=$rsync_rc" "rsync $SRC/ -> $DEST/ failed rc=$rsync_rc

stderr:
$(cat "$err_tmp")"
fi
rm -f "$err_tmp"

exit "$rsync_rc"
