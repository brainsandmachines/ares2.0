#!/bin/bash
# Daily backup of the AIRCC results/models tree to Botero, pulled through the local
# sshfs AIRCC mount (no ssh in the rsync itself). Run from a Botero cron.
#
# Install (Botero crontab -e):
#   0 3 * * * /home/tomer_a/Documents/ares/aircc/aircc_job_manager/scripts/backup_aircc_models.sh >> /mnt/data/robustness_models/aircc_models/backup.log 2>&1
#
# Permissions: --no-perms/--no-owner/--no-group drop the sshfs-reported perms so
# files land owned by your Botero user; the dest root is created mode 0755.

set -euo pipefail

# --- CONFIRM THESE PATHS ---
# Local sshfs mount of the AIRCC results/models dir, e.g.:
#   ~/aircc_mount/shared/cycle2_bgu_golan_prj/ashtomer/ares/results/models
SRC="${AIRCC_MOUNT:-$HOME/aircc_mount/ashtomer/ares/results/models}"
DEST="/mnt/data/robustness_models/aircc_models"

if [[ ! -d "$SRC" ]]; then
    echo "[backup] ERROR: source not mounted/missing: $SRC" >&2
    echo "[backup] is the sshfs AIRCC mount up?" >&2
    exit 1
fi

mkdir -p -m 0755 "$DEST"

echo "[backup] $(date -Is) rsync $SRC/ -> $DEST/"
rsync -rt --no-perms --no-owner --no-group --partial --info=stats2 \
    "$SRC/" "$DEST/"
echo "[backup] $(date -Is) done"
