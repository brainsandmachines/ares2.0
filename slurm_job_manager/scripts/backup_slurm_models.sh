#!/bin/bash
# Weekly archive of the Slurm cluster's results/models tree onto Botero (/mnt/data4t).
#
# CURATED, not byte-for-byte: the remote tree is ~3.7TB, of which ~2.7TB is intermediate
# checkpoint-N.pth.tar. Those are excluded, leaving ~0.9TB of the files that are actually
# worth keeping -- last / model_best / model_best_adv / periodic/epoch_* plus every log,
# hydra config, csv and plot. /mnt/data4t also carries the growing AIRCC archive, so the
# full tree simply does not fit; the curated one does, twice over (here and on the QNAP).
#
# This job does NOT touch the QNAP. scripts/mirror_archives_to_qnap.sh runs daily and
# picks up whatever this leaves on /mnt/data4t.
#
# APPEND-ONLY: no --delete, ever. The group filesystem is 79% full and gets cleared; a
# --delete pass after that would erase the only remaining copy.
#
# Install (Botero crontab -e):
#   0 9 * * 0 /home/tomer_a/Documents/ares/slurm_job_manager/scripts/backup_slurm_models.sh >> /mnt/data4t/slurm_archive/backup.log 2>&1
#
# Exit codes: 0 = backup ran and succeeded, 75 = nothing to do (not due yet, or another run
# holds the lock), anything else = a real failure (already emailed).

set -u -o pipefail

# --- CONFIRM THESE PATHS ---
SLURM_HOST="${SLURM_SSH_HOST:-slurm}"
# The REAL path, not the results/models symlink in the repo: rsync's handling of a symlinked
# source is a footgun, and this path is what the symlink resolves to anyway.
REMOTE_ROOT="${SJM_REMOTE_RESULTS:-/groups/golan_neurogroup/bml_group/tomerash/advmodels/results}"
DEST_ROOT="${SJM_BACKUP_DEST:-/mnt/data4t/slurm_archive}"
# Live job DB, read over ssh (immutable=1). Deliberately NOT via ~/slurm_mount -- that sshfs
# mount is frequently not up, and this script already needs ssh to the login node.
REMOTE_DB="${SJM_REMOTE_DB:-/home/ashtomer/projects/ares/slurm_job_manager/jobs.sqlite}"
REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
LOCK_FILE="${SJM_BACKUP_LOCK:-$DEST_ROOT/.backup.lock}"
STAMP_FILE="${SJM_BACKUP_STAMP:-$DEST_ROOT/.backup.started}"
# Written when a run STARTS an rsync and never removed -- this paces the backup, so a failed or
# killed pass waits its turn instead of retrying on the next cron tick.
ATTEMPT_FILE="${SJM_BACKUP_ATTEMPT:-$DEST_ROOT/.backup.attempted}"
# The cron is weekly; this guards a manual re-run or a doubled cron line, and is low enough that
# a run killed early in the week can simply be started again by hand.
MIN_INTERVAL_HOURS="${SJM_BACKUP_MIN_INTERVAL_HOURS:-120}"
# ~0.9TB at the measured ~67MB/s is 4-5h; 24h absorbs a slow week without ever reaching the
# next weekly tick.
RSYNC_TIMEOUT="${SJM_BACKUP_RSYNC_TIMEOUT:-24h}"
STALE_HOURS="${SJM_BACKUP_STALE_HOURS:-30}"
# Refuse to start a ~0.9TB pull with less than this free on /mnt/data4t (shared with the AIRCC
# archive, which is still growing).
MIN_FREE_GB="${SJM_BACKUP_MIN_FREE_GB:-200}"
DRY_RUN=0
EXIT_SKIPPED=75

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p -m 0755 "$DEST_ROOT"

notify() {
    local subject="$1" body="$2"
    local dedup="${3:-}" urgent="${4:-}"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python3 - "$subject" "$body" "$dedup" "$urgent" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body, dedup, urgent = sys.argv[1:5]
emailer = make_emailer(source="sjm.backup")
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body, dedup_key=dedup or None, urgent=bool(urgent))
PYEOF
}

# --- 1. is a backup due? ---
# Paced off the last ATTEMPT, not the last success (same rule as the AIRCC backup).
if [[ -f "$ATTEMPT_FILE" && "$DRY_RUN" -eq 0 ]]; then
    since_h=$(( ( $(date +%s) - $(stat -c %Y "$ATTEMPT_FILE") ) / 3600 ))
    if [[ "$since_h" -lt "$MIN_INTERVAL_HOURS" ]]; then
        echo "[backup] $(date -Is) SKIP: last attempt was ${since_h}h ago, next due in $(( MIN_INTERVAL_HOURS - since_h ))h"
        exit "$EXIT_SKIPPED"
    fi
fi

# --- 2. single-instance lock ---
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    held_since="$(cat "$STAMP_FILE" 2>/dev/null || echo "unknown")"
    holders="$(fuser "$LOCK_FILE" 2>/dev/null | tr -s ' ')"
    stale=0
    if [[ -f "$STAMP_FILE" ]]; then
        age_h=$(( ( $(date +%s) - $(stat -c %Y "$STAMP_FILE") ) / 3600 ))
        [[ "$age_h" -ge "$STALE_HOURS" ]] && stale=1
    else
        age_h="unknown"
        stale=1
    fi

    if [[ "$stale" -eq 1 ]]; then
        echo "[backup] $(date -Is) STALE LOCK: held ${age_h}h (since $held_since), holders:${holders:- none}" >&2
        notify "[sjm] slurm archive lock held for ${age_h}h -- backups are not running" \
"$LOCK_FILE has been held since $held_since (${age_h}h), so this backup SKIPped.
Every run will keep skipping until the holder dies.

holding pids:${holders:- (none found)}

Usually a hung ssh/rsync -- SIGKILL clears it:
  fuser -k $LOCK_FILE" "" urgent
        exit 1
    fi

    echo "[backup] $(date -Is) SKIP: another backup run holds $LOCK_FILE (${age_h}h, since $held_since)" >&2
    exit "$EXIT_SKIPPED"
fi

date -Is > "$STAMP_FILE"
trap 'rm -f "$STAMP_FILE"' EXIT

# --- 3. preflight: reachable source, room to land it ---
if ! ssh -o BatchMode=yes -o ConnectTimeout=30 "$SLURM_HOST" test -d "$REMOTE_ROOT/models"; then
    echo "[backup] ERROR: cannot reach $SLURM_HOST, or $REMOTE_ROOT/models is gone" >&2
    notify "[sjm] slurm archive source unreachable" \
"ssh $SLURM_HOST test -d $REMOTE_ROOT/models failed.
Either the cluster is unreachable or the results tree no longer exists." "" urgent
    exit 1
fi

free_gb=$(df -BG --output=avail "$DEST_ROOT" | tail -1 | tr -dc '0-9')
if [[ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]]; then
    echo "[backup] ERROR: only ${free_gb}GB free on $DEST_ROOT (need ${MIN_FREE_GB}GB)" >&2
    notify "[sjm] slurm archive: not enough free space" \
"$DEST_ROOT has ${free_gb}GB free, below the ${MIN_FREE_GB}GB floor.
/mnt/data4t also carries the growing AIRCC archive. Refusing to start the pull." "" urgent
    exit 1
fi

# --- 4. exclude models that are training right now ---
# sjm's jobs.model_name IS the path of the model dir relative to results/models (e.g.
# 'swin_b/l1_1_init1'), so a running row maps straight to an rsync exclude.
#
# Read into a variable, NOT `mapfile < <(...)`: a process substitution's exit status is not
# propagated, so a failed ssh or unreadable DB would look like "nothing is running" and this
# would happily pull half-written checkpoints. Fail closed instead.
if ! running_raw="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$SLURM_HOST" \
    "python3 -c \"
import sqlite3
con = sqlite3.connect('file:$REMOTE_DB?immutable=1', uri=True)
for (name,) in con.execute(\\\"SELECT model_name FROM jobs WHERE status='running'\\\"):
    print(name)
\"")"; then
    echo "[backup] ERROR: cannot read the sjm job DB at $SLURM_HOST:$REMOTE_DB" >&2
    notify "[sjm] slurm archive: cannot read job DB" \
"$SLURM_HOST:$REMOTE_DB could not be read, so the running-model exclusions could not be built.
Refusing to sync rather than pulling mid-training checkpoints." "" urgent
    exit 1
fi

running_excludes=()
if [[ -n "$running_raw" ]]; then
    while IFS= read -r name; do
        [[ -z "$name" ]] && continue
        running_excludes+=("--exclude=/$name/")
        echo "[backup] skipping running model: $name"
    done <<<"$running_raw"
fi
echo "[backup] skipping ${#running_excludes[@]} running model dir(s)"

# --- 5. rsync ---
benign_pat='^(file has vanished: .*|rsync warning: .*|rsync error: some files/attrs were not transferred.*|[[:space:]]*)$'

common_args=(
    -rt --no-perms --no-owner --no-group --partial --info=stats2
    # The 2.7TB of intermediate checkpoints. Anchored on the leading 'checkpoint-' so it can
    # never match model_best*/last/epoch_* .
    --exclude='checkpoint-[0-9]*.pth.tar'
    --exclude='tmp.pth.tar'
    # Protect this script's own control files, which live in $DEST_ROOT rather than under $SRC.
    --filter='protect /backup.log' --filter='protect /.backup.lock'
    --filter='protect /.backup.started' --filter='protect /.backup.attempted'
)
[[ "$DRY_RUN" -eq 1 ]] && common_args+=(--dry-run --itemize-changes)

run_leg() {  # run_leg <name> <remote-subdir> <dest-subdir> [extra rsync args...]
    local name="$1" remote_sub="$2" dest_sub="$3"; shift 3
    local src="$SLURM_HOST:$REMOTE_ROOT/$remote_sub" dest="$DEST_ROOT/$dest_sub"
    local err_tmp rc=0 real_errs

    [[ "$DRY_RUN" -eq 0 ]] && mkdir -p "$dest"
    echo "[backup] $(date -Is) rsync $src/ -> $dest/"
    err_tmp="$(mktemp "${TMPDIR:-/tmp}/sjm_rsync_err.XXXXXX")"
    timeout --kill-after=5m "$RSYNC_TIMEOUT" \
        rsync "${common_args[@]}" "$@" "$src/" "$dest/" 2>"$err_tmp" || rc=$?
    cat "$err_tmp" >&2

    if [[ "$rc" -eq 0 ]]; then
        echo "[backup] $(date -Is) $name done"
    elif [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
        echo "[backup] $(date -Is) ERROR: $name exceeded $RSYNC_TIMEOUT and was killed (rc=$rc)" >&2
    elif [[ "$rc" -eq 23 || "$rc" -eq 24 ]]; then
        real_errs="$(grep -vE "$benign_pat" "$err_tmp")"
        if [[ -n "$real_errs" ]]; then
            echo "[backup] $(date -Is) ERROR: $name failed rc=$rc (non-vanished errors on stderr)" >&2
        else
            echo "[backup] $(date -Is) note: $name rc=$rc, only files that vanished mid-transfer"
            echo "[backup] $(date -Is) $name done"
            rc=0
        fi
    else
        echo "[backup] $(date -Is) ERROR: $name failed rc=$rc" >&2
    fi

    if [[ "$rc" -ne 0 ]]; then
        notify "[sjm] slurm archive: $name failed rc=$rc" "rsync $src/ -> $dest/ failed rc=$rc

stderr:
$(cat "$err_tmp")" "" urgent
    fi
    rm -f "$err_tmp"
    return "$rc"
}

[[ "$DRY_RUN" -eq 0 ]] && date -Is > "$ATTEMPT_FILE"   # starts the pacing clock

overall_rc=0
# 'for deletion' -- 422GB you already marked; quoted as a single argv element (the name has a
# space in it). Excluded here only, not from models_failed.
run_leg models models models \
    --exclude='/for deletion/' "${running_excludes[@]}" || overall_rc=$?
# 19GB, and no run writes into it while training, so no exclusions needed.
run_leg models_failed models_failed models_failed || overall_rc=$?

exit "$overall_rc"
