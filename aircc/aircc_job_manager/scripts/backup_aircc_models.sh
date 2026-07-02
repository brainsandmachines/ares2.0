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

mkdir -p "$(dirname "$MONITOR_LOG")"

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

mkdir -p -m 0755 "$DEST"

echo "[backup] $(date -Is) rsync $SRC/ -> $DEST/"
rsync_rc=0
rsync -rt --no-perms --no-owner --no-group --partial --info=stats2 \
    "$SRC/" "$DEST/" || rsync_rc=$?
if [[ "$rsync_rc" -eq 0 ]]; then
    echo "[backup] $(date -Is) done"
else
    echo "[backup] $(date -Is) ERROR: rsync failed rc=$rsync_rc" >&2
fi

monitor_rc=0
run_monitor "$rsync_rc" || monitor_rc=$?
if [[ "$rsync_rc" -ne 0 ]]; then
    exit "$rsync_rc"
fi
exit "$monitor_rc"
