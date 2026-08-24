#!/bin/bash
# Daily Botero-local mirror of both model archives onto the QNAP share.
#
#   /mnt/data4t/aircc_archive/  ->  /mnt/botero/aircc_archive/
#   /mnt/data4t/slurm_archive/  ->  /mnt/botero/slurm_archive/
#
# This job reaches NO cluster. The two pulls that fill /mnt/data4t are separate and
# independent of it:
#   - AIRCC: aircc/aircc_job_manager/scripts/backup_aircc_models.sh (03:00 cron, paced 72h)
#   - Slurm: slurm_job_manager/scripts/backup_slurm_models.sh       (weekly cron)
# It deliberately does not wait for or chain off either -- it mirrors whatever is on
# /mnt/data4t at 06:00, and anything a pull was mid-writing is corrected the next day.
#
# APPEND-ONLY, like the archives it mirrors: no --delete, ever. Both clusters delete their
# results trees, and a --delete pass after that would faithfully erase the only copies.
#
# Install (Botero crontab -e):
#   0 6 * * * /home/tomer_a/Documents/ares/scripts/mirror_archives_to_qnap.sh >> /mnt/data4t/qnap_mirror.log 2>&1
#
# Exit codes: 0 = every present leg mirrored, 75 = nothing to do (another run holds the
# lock), anything else = a real failure (already emailed).

set -u -o pipefail

# --- CONFIRM THESE PATHS ---
SRC_ROOT="${QNAP_MIRROR_SRC_ROOT:-/mnt/data4t}"
# The CIFS share (//132.72.128.88/backup/botero). MUST be a real mountpoint -- see the guard below.
QNAP_ROOT="${QNAP_MIRROR_DEST_ROOT:-/mnt/botero}"
# Archive dirs to mirror, relative to both roots. A missing source is skipped, not an error,
# so this job works on day one, before the first slurm pull has run.
ARCHIVES=(${QNAP_MIRROR_ARCHIVES:-aircc_archive slurm_archive})
REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
# The lock lives on LOCAL disk, never on the CIFS dest: flock over CIFS is not dependable,
# and a lock we cannot trust is worse than no lock.
LOCK_FILE="${QNAP_MIRROR_LOCK:-/tmp/ares_qnap_mirror.lock}"
STAMP_FILE="${QNAP_MIRROR_STAMP:-/tmp/ares_qnap_mirror.started}"
# Per-leg cap. The AIRCC leg's first full pass moves ~1.8TB over LAN; 12h is generous for that
# and still leaves the next daily run a clear field.
RSYNC_TIMEOUT="${QNAP_MIRROR_RSYNC_TIMEOUT:-12h}"
# A lock older than this is wedged, not busy (the timeout above would have killed a live run).
STALE_HOURS="${QNAP_MIRROR_STALE_HOURS:-30}"
DRY_RUN=0
EXIT_SKIPPED=75

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

notify() {
    local subject="$1" body="$2"
    local dedup="${3:-}" urgent="${4:-}"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python3 - "$subject" "$body" "$dedup" "$urgent" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body, dedup, urgent = sys.argv[1:5]
emailer = make_emailer(source="qnap.mirror")
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body, dedup_key=dedup or None, urgent=bool(urgent))
PYEOF
}

# --- 1. mount guard (the important one) ---
# If the CIFS share is not mounted, $QNAP_ROOT is an ordinary directory on / -- which has ~130GB
# free against ~2.7TB of archive. An unguarded rsync would fill the root filesystem before dawn.
# Two independent checks: the kernel's view (mountpoint) and a sentinel that only exists on the
# share (a mountpoint test alone would pass on a stale/other mount).
if ! mountpoint -q "$QNAP_ROOT"; then
    echo "[mirror] $(date -Is) ERROR: $QNAP_ROOT is not a mountpoint -- refusing to sync" >&2
    notify "[qnap] mirror aborted: $QNAP_ROOT not mounted" \
"$QNAP_ROOT is not a mountpoint, so it is a plain directory on the root filesystem.
Syncing into it would fill / (about 130GB free vs ~2.7TB of archive), so this run refused.

Remount the QNAP share and the next run picks up where this one stopped." \
        "qnap-mirror-not-mounted" urgent
    exit 1
fi

# The sentinel is the AIRCC archive dir itself: it is the tree you created on the share by hand,
# it is never removed, and it does not exist on the local-disk fallback path.
SENTINEL="${QNAP_MIRROR_SENTINEL:-$QNAP_ROOT/aircc_archive}"
if [[ ! -d "$SENTINEL" ]]; then
    echo "[mirror] $(date -Is) ERROR: sentinel $SENTINEL missing -- refusing to sync" >&2
    notify "[qnap] mirror aborted: sentinel missing" \
"$QNAP_ROOT is mounted but $SENTINEL does not exist.
That usually means a different share is mounted there, or the archive dir was removed.
Refusing to sync rather than writing ~2.7TB into the wrong place." \
        "qnap-mirror-no-sentinel" urgent
    exit 1
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
        echo "[mirror] $(date -Is) STALE LOCK: held ${age_h}h (since $held_since), holders:${holders:- none}" >&2
        notify "[qnap] mirror lock held for ${age_h}h -- the QNAP copy is not updating" \
"$LOCK_FILE has been held since $held_since (${age_h}h), so this run SKIPped.
Every run will keep skipping until the holder dies.

holding pids:${holders:- (none found)}

A holder stuck on the CIFS share is usually an rsync in D state; the share is mounted 'soft',
so it should return an IO error rather than hang, but a wedged one needs SIGKILL:
  fuser -k $LOCK_FILE" \
            "" urgent
        exit 1
    fi

    echo "[mirror] $(date -Is) SKIP: another mirror run holds $LOCK_FILE (${age_h}h, since $held_since)" >&2
    exit "$EXIT_SKIPPED"
fi

date -Is > "$STAMP_FILE"
trap 'rm -f "$STAMP_FILE"' EXIT

# --- 3. mirror each archive ---
# Lines that are expected when mirroring a tree an rsync pull may still be writing into.
benign_pat='^(file has vanished: .*|rsync warning: .*|rsync error: some files/attrs were not transferred.*|[[:space:]]*)$'

overall_rc=0
for archive in "${ARCHIVES[@]}"; do
    src="$SRC_ROOT/$archive"
    dest="$QNAP_ROOT/$archive"

    if [[ ! -d "$src" ]]; then
        echo "[mirror] $(date -Is) skip $archive: $src does not exist yet"
        continue
    fi

    echo "[mirror] $(date -Is) rsync $src/ -> $dest/"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$dest" || { overall_rc=1; continue; }
    elif [[ ! -d "$dest" ]]; then
        echo "[mirror] (dry-run) $dest does not exist yet; a real run would create it"
        continue
    fi

    rsync_args=(
        -rt --no-perms --no-owner --no-group --partial --info=stats2
        # The AIRCC pull's own control files live in its archive root, not under its source --
        # they describe a Botero-side job and have no business on the QNAP.
        --exclude=/models/backup.log --exclude=/models/.backup.lock
        --exclude=/models/.backup.started --exclude=/models/.backup.attempted
        --exclude=/backup.log --exclude=/mirror_*.log
        # Already staged for deletion on the AIRCC side; no reason to spend 96GB of share on it.
        --exclude=/_pending_delete_*/
    )
    [[ "$DRY_RUN" -eq 1 ]] && rsync_args+=(--dry-run --itemize-changes)

    err_tmp="$(mktemp "${TMPDIR:-/tmp}/qnap_mirror_err.XXXXXX")"
    rc=0
    timeout --kill-after=5m "$RSYNC_TIMEOUT" \
        rsync "${rsync_args[@]}" "$src/" "$dest/" 2>"$err_tmp" || rc=$?
    cat "$err_tmp" >&2

    if [[ "$rc" -eq 0 ]]; then
        echo "[mirror] $(date -Is) $archive done"
    elif [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
        echo "[mirror] $(date -Is) ERROR: $archive exceeded $RSYNC_TIMEOUT and was killed (rc=$rc)" >&2
    elif [[ "$rc" -eq 23 || "$rc" -eq 24 ]]; then
        # A pull may be rewriting the source underneath us; vanished files are expected churn,
        # anything else on stderr is not.
        real_errs="$(grep -vE "$benign_pat" "$err_tmp")"
        if [[ -n "$real_errs" ]]; then
            echo "[mirror] $(date -Is) ERROR: $archive failed rc=$rc (non-vanished errors on stderr)" >&2
        else
            echo "[mirror] $(date -Is) note: $archive rc=$rc, only files that vanished mid-transfer"
            echo "[mirror] $(date -Is) $archive done"
            rc=0
        fi
    else
        echo "[mirror] $(date -Is) ERROR: $archive failed rc=$rc" >&2
    fi

    if [[ "$rc" -ne 0 ]]; then
        notify "[qnap] mirror of $archive failed rc=$rc" "rsync $src/ -> $dest/ failed rc=$rc

stderr:
$(cat "$err_tmp")" "" urgent
        overall_rc="$rc"
    fi
    rm -f "$err_tmp"
    # Deliberately no `break`: one failing leg must not cost the other its daily pass.
done

exit "$overall_rc"
