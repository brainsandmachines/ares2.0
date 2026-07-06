#!/bin/bash
# Daily backup of the AIRCC results/models tree to Botero, pulled through the local
# sshfs AIRCC mount (no ssh in the rsync itself). Run from a Botero cron.
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
MONITOR_LOG="${AIRCC_MONITOR_LOG:-$REPO_ROOT/aircc/aircc_job_manager/logs/daily_monitor.log}"
LOCK_FILE="${AIRCC_BACKUP_LOCK:-$DEST/.backup.lock}"

mkdir -p "$(dirname "$MONITOR_LOG")"
mkdir -p -m 0755 "$DEST"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[backup] $(date -Is) SKIP: another backup run already holds $LOCK_FILE" >&2
    exit 0
fi

run_monitor() {
    local backup_rc="$1"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
        python -m aircc.aircc_job_manager.daily_monitor \
            --backup-log "$DEST/backup.log" \
            --backup-rc "$backup_rc" \
            >> "$MONITOR_LOG" 2>&1
}

if [[ ! -d "$SRC" ]]; then
    echo "[backup] ERROR: source not mounted/missing: $SRC" >&2
    echo "[backup] is the sshfs AIRCC mount up?" >&2
    run_monitor 1
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
rm -f "$err_tmp"

monitor_rc=0
run_monitor "$rsync_rc" || monitor_rc=$?
if [[ "$rsync_rc" -ne 0 ]]; then
    exit "$rsync_rc"
fi
exit "$monitor_rc"
