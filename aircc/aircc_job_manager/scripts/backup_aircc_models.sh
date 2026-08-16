#!/bin/bash
# AIRCC check-in, run nightly from a Botero cron but with two different cadences:
#
#   (1) status check -- EVERY run. Runs the aircc-status skill's status script and
#       emails if the running count doesn't match squeue's live expected count
#       (2 models per RUNNING aircc_jm task) or any job has failed. Cheap, and
#       these are the alerts worth having daily.
#   (2) rsync backup -- at most every MIN_INTERVAL_HOURS (default 72h, i.e. every
#       3 days). Backs up the AIRCC results/models tree to Botero, pulled through
#       the local sshfs AIRCC mount (no ssh in the rsync itself), emailing only if
#       the rsync itself fails. A full pass moves ~450GB over sshfs at a few MB/s
#       and can run well over a day, so nightly attempts were mostly no-ops that
#       collided with the previous night's run.
#
# Exit codes: 0 = backup ran and succeeded, 75 = nothing to do (not due yet, or a
# previous backup is still running), anything else = a real failure. The caller
# passes this to daily_monitor as --backup-rc, and 75 tells the monitor not to
# validate a backup block that this run did not produce.
#
# Install (Botero crontab -e) -- nightly; the every-3-days part is enforced here,
# not in the cron expression, so the daily status check and daily_monitor's DB
# health check keep running on the off days:
#   0 3 * * * /home/tomer_a/Documents/ares/aircc/aircc_job_manager/scripts/backup_aircc_models.sh >> /mnt/data/robustness_models/aircc_models/backup.log 2>&1
#
# Permissions: --no-perms/--no-owner/--no-group drop the sshfs-reported perms so
# files land owned by your Botero user; the dest root is created mode 0755.

set -u -o pipefail

# --- CONFIRM THESE PATHS ---
# Local sshfs mount of the AIRCC results/models dir, e.g.:
#   ~/aircc_mount/shared/cycle2_bgu_golan_prj/ashtomer/ares/results/models
SRC="${AIRCC_MOUNT:-$HOME/aircc_mount/ashtomer/ares/results/models}"
DEST="${AIRCC_BACKUP_DEST:-/mnt/data/robustness_models/aircc_models}"
REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
STATUS_SCRIPT="${AIRCC_STATUS_SCRIPT:-$HOME/.claude/skills/aircc-status/scripts/aircc_status.py}"
LOCK_FILE="${AIRCC_BACKUP_LOCK:-$DEST/.backup.lock}"
# Written when a run takes the lock, removed when it finishes. A skipped run compares its age
# against STALE_HOURS to tell "the last run is still going" from "a run is wedged".
STAMP_FILE="${AIRCC_BACKUP_STAMP:-$DEST/.backup.started}"
# Written when a run STARTS an rsync and never removed -- this is what paces the backup, so a
# failed or killed pass still waits its turn instead of retrying every night.
ATTEMPT_FILE="${AIRCC_BACKUP_ATTEMPT:-$DEST/.backup.attempted}"
MIN_INTERVAL_HOURS="${AIRCC_BACKUP_MIN_INTERVAL_HOURS:-72}"
# Hard cap on the rsync itself. The full pass moves ~450GB over sshfs at ~3.5MB/s and has taken
# well over a day, so this has to be generous -- but it must stay under MIN_INTERVAL_HOURS with
# room to spare, or a wedged run would still hold the lock when the next backup is due.
RSYNC_TIMEOUT="${AIRCC_BACKUP_RSYNC_TIMEOUT:-60h}"
# A lock older than this is not "still going" (the timeout above would have killed it) -- it is
# wedged. Keep between RSYNC_TIMEOUT and MIN_INTERVAL_HOURS.
STALE_HOURS="${AIRCC_BACKUP_STALE_HOURS:-66}"
EXIT_SKIPPED=75

mkdir -p -m 0755 "$DEST"

notify() {
    local subject="$1" body="$2"
    local dedup="${3:-}" urgent="${4:-}"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python3 - "$subject" "$body" "$dedup" "$urgent" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body, dedup, urgent = sys.argv[1:5]
emailer = make_emailer(source="aircc.backup")
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body, dedup_key=dedup or None, urgent=bool(urgent))
PYEOF
}

# --- 1. aircc-status check (every run, including nights with no backup) ---
# Deliberately ahead of the lock: this costs seconds, and a backup pass can hold
# the lock for more than a day, which must not silence the daily status alerts.
#
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

$status_out" "aircc-status-check-failed"
elif [[ -z "$expected_n" ]]; then
    # squeue SSH check was unavailable -- can't compare against a live expected
    # count, so only alert on failures, don't fabricate a running-count mismatch.
    if [[ "${failed_n:-0}" -ne 0 ]]; then
        echo "[status] $(date -Is) ALERT: failed=${failed_n:-0} (expected_running check unavailable)" >&2
        notify "[aircc] status alert: failed=${failed_n:-0}" "$status_out" \
            "aircc-status:failed=${failed_n:-0}"
    else
        echo "[status] $(date -Is) ok: running=${running_n:-0} failed=0 (expected_running check unavailable)"
    fi
elif [[ "$check_state" != "[OK]" || "${failed_n:-0}" -ne 0 ]]; then
    echo "[status] $(date -Is) ALERT: running=${running_n:-0} (expected $expected_n) failed=${failed_n:-0}" >&2
    notify "[aircc] status alert: running=${running_n:-0} (expected $expected_n) failed=${failed_n:-0}" \
        "$status_out" "aircc-status:running=${running_n:-0}:expected=${expected_n}:failed=${failed_n:-0}"
else
    echo "[status] $(date -Is) ok: running=$running_n (expected $expected_n) failed=0"
fi

# --- 2. is a backup due? ---
# Paced off the last ATTEMPT, not the last success: a pass that fails or hits the timeout waits its
# turn like any other, instead of retrying every night. Missing stamp => due (first run ever).
if [[ -f "$ATTEMPT_FILE" ]]; then
    since_h=$(( ( $(date +%s) - $(stat -c %Y "$ATTEMPT_FILE") ) / 3600 ))
    if [[ "$since_h" -lt "$MIN_INTERVAL_HOURS" ]]; then
        echo "[backup] $(date -Is) SKIP: last attempt was ${since_h}h ago, next due in $(( MIN_INTERVAL_HOURS - since_h ))h"
        exit "$EXIT_SKIPPED"
    fi
fi

# --- 3. single-instance lock ---
# On 2026-07-27 a run wedged here for a week: the aircc sshfs mount was remounted underneath it, so
# its rsync sat in uninterruptible D state on the abandoned FUSE connection, ignoring SIGKILL. rsync
# inherits fd 9, and an flock belongs to the open file description, so the lock outlived the script
# and every night after that silently SKIPped. A skip is normal; a skip on an old lock is not.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    held_since="$(cat "$STAMP_FILE" 2>/dev/null || echo "unknown")"
    holders="$(fuser "$LOCK_FILE" 2>/dev/null | tr -s ' ')"
    stale=0
    if [[ -f "$STAMP_FILE" ]]; then
        age_h=$(( ( $(date +%s) - $(stat -c %Y "$STAMP_FILE") ) / 3600 ))
        [[ "$age_h" -ge "$STALE_HOURS" ]] && stale=1
    else
        # Lock held with no stamp: a pre-hardening run, or the stamp was lost. Treat as suspect.
        age_h="unknown"
        stale=1
    fi

    if [[ "$stale" -eq 1 ]]; then
        echo "[backup] $(date -Is) STALE LOCK: held ${age_h}h (since $held_since), holders:${holders:- none}" >&2
        notify "[aircc] backup lock held for ${age_h}h -- backups are not running" \
"$LOCK_FILE has been held since $held_since (${age_h}h), so this backup SKIPped.
Every run will keep skipping until the holder dies.

holding pids:${holders:- (none found)}

If a holder is in D state it is stuck on a dead sshfs connection and SIGKILL will not work.
Find the stale FUSE connection (one with no matching mount in /proc/self/mountinfo and
waiting > 0) and abort just that one:

  ls -l /sys/fs/fuse/connections/
  for c in /sys/fs/fuse/connections/*/; do echo \"\$c waiting=\$(cat \$c/waiting)\"; done
  echo 1 > /sys/fs/fuse/connections/<stale-id>/abort
" "" urgent
        exit 1
    fi

    echo "[backup] $(date -Is) SKIP: another backup run already holds $LOCK_FILE (${age_h}h, since $held_since)" >&2
    exit "$EXIT_SKIPPED"
fi

date -Is > "$STAMP_FILE"
trap 'rm -f "$STAMP_FILE"' EXIT

# --- 4. rsync backup ---
if [[ ! -d "$SRC" ]]; then
    echo "[backup] ERROR: source not mounted/missing: $SRC" >&2
    echo "[backup] is the sshfs AIRCC mount up?" >&2
    notify "[aircc] backup source not mounted" "source not mounted/missing: $SRC
is the sshfs AIRCC mount up?" "" urgent
    exit 1
fi

echo "[backup] $(date -Is) rsync $SRC/ -> $DEST/"
date -Is > "$ATTEMPT_FILE"   # starts the MIN_INTERVAL_HOURS clock for the next run
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
# timeout caps a run that is merely slow or stalled on a live connection. It cannot kill an rsync
# wedged in D state on a dead FUSE connection -- the stale-lock alert above is what catches that.
# --kill-after escalates to SIGKILL if the rsync ignores SIGTERM.
timeout --kill-after=5m "$RSYNC_TIMEOUT" \
    rsync -rt --no-perms --no-owner --no-group --partial --delete-after --delete-excluded \
    --filter='protect /backup.log' --filter='protect /.backup.lock' \
    --filter='protect /.backup.started' --filter='protect /.backup.attempted' \
    --exclude='checkpoint-*.pth.tar' --info=stats2,progress2 \
    "$SRC/" "$DEST/" 2>"$err_tmp" || rsync_rc=$?
cat "$err_tmp" >&2   # keep the stderr lines in the log

# Lines that are expected when backing up a live training tree. Anything on
# rsync's stderr that does NOT match one of these is treated as a real error.
benign_pat='^(file has vanished: .*|rsync warning: .*|rsync error: some files/attrs were not transferred.*|[[:space:]]*)$'

if [[ "$rsync_rc" -eq 0 ]]; then
    echo "[backup] $(date -Is) done"
elif [[ "$rsync_rc" -eq 124 || "$rsync_rc" -eq 137 ]]; then
    # 124 = timeout fired, 137 = had to escalate to SIGKILL.
    echo "[backup] $(date -Is) ERROR: rsync exceeded ${RSYNC_TIMEOUT} and was killed (rc=$rsync_rc)" >&2
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
$(cat "$err_tmp")" "" urgent
fi
rm -f "$err_tmp"

exit "$rsync_rc"
