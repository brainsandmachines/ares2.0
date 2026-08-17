#!/bin/bash
# AIRCC check-in, run nightly from a Botero cron but with two different cadences:
#
#   (1) status check -- EVERY run. Runs the aircc-status skill's status script and
#       emails if the running count doesn't match squeue's live expected count
#       (2 models per RUNNING aircc_jm task) or any job has failed. Cheap, and
#       these are the alerts worth having daily.
#   (2) rsync backup -- at most every MIN_INTERVAL_HOURS (default 72h, i.e. every
#       3 days). Archives the AIRCC results/models tree to Botero over direct ssh,
#       emailing only if the rsync itself fails. A full pass moves ~1.9TB at
#       ~25MB/s and can run most of a day, so nightly attempts were mostly
#       no-ops that collided with the previous night's run.
#
#       This is an ARCHIVE, not a mirror: AIRCC deletes the results tree in late
#       August 2026, and it holds every checkpoint rather than just the best/last.
#       See the rsync block below for why nothing is ever deleted from the
#       destination and why mid-training dirs are skipped.
#
# Exit codes: 0 = backup ran and succeeded, 75 = nothing to do (not due yet, or a
# previous backup is still running), anything else = a real failure. The caller
# passes this to daily_monitor as --backup-rc, and 75 tells the monitor not to
# validate a backup block that this run did not produce.
#
# Install (Botero crontab -e) -- nightly; the every-3-days part is enforced here,
# not in the cron expression, so the daily status check and daily_monitor's DB
# health check keep running on the off days:
#   0 3 * * * /home/tomer_a/Documents/ares/aircc/aircc_job_manager/scripts/backup_aircc_models.sh >> /mnt/data4t/aircc_archive/models/backup.log 2>&1
#
# Permissions: --no-perms/--no-owner/--no-group drop the remote-reported perms so
# files land owned by your Botero user; the dest root is created mode 0755.

set -u -o pipefail

# --- CONFIRM THESE PATHS ---
# Pulled straight over ssh rather than through the sshfs mount: direct ssh measured 9.7MB/s vs
# 3.7MB/s through the mount (measured 2026-08-17), which matters now that the intermediate
# checkpoints are included too. Real rsync throughput runs ~25MB/s; the 9.7 figure was a dd probe.
# REMOTE_DIR is the same path without the host, for the pre-flight existence test.
AIRCC_HOST="${AIRCC_HOST:-aircc}"
REMOTE_DIR="${AIRCC_REMOTE_DIR:-/shared/cycle2_bgu_golan_prj/ashtomer/ares/results/models}"
SRC="${AIRCC_SRC:-$AIRCC_HOST:$REMOTE_DIR}"
# The archive lives on /mnt/data4t (3.6T): the full tree incl. every checkpoint is ~1.9TB, which
# does not fit alongside everything else on /mnt/data. Deletions are never propagated here.
DEST="${AIRCC_BACKUP_DEST:-/mnt/data4t/aircc_archive/models}"
# Job DB, read (immutable) only to find which models are mid-training. Still via the sshfs mount --
# it is a 12KB file and the status check above already depends on the mount being up.
AIRCC_DB_PATH="${AIRCC_DB_PATH:-$HOME/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/aircc_jobs.sqlite}"
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
# Hard cap on the rsync itself. A first/catch-all pass moves up to ~1.9TB at ~25MB/s (~21h), so this
# has to be generous -- but it must stay under MIN_INTERVAL_HOURS with room to spare, or a wedged run
# would still hold the lock when the next backup is due.
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
# The rsync now goes over direct ssh, so that exact FUSE wedge can no longer happen to it -- but a
# hung ssh holds the lock just as effectively, so this check still earns its place.
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

A holder in S state is usually a hung ssh/rsync -- SIGKILL clears it.

If a holder is in D state it is stuck on a dead sshfs connection and SIGKILL will not work.
The rsync itself no longer reads through the mount, but the status check does. Find the stale
FUSE connection (one with no matching mount in /proc/self/mountinfo and waiting > 0) and abort
just that one:

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
if ! ssh -o BatchMode=yes -o ConnectTimeout=30 "$AIRCC_HOST" test -d "$REMOTE_DIR"; then
    echo "[backup] ERROR: cannot reach $AIRCC_HOST, or $REMOTE_DIR is gone" >&2
    notify "[aircc] backup source unreachable" "ssh $AIRCC_HOST test -d $REMOTE_DIR failed.
Either the cluster is unreachable or the results tree no longer exists." "" urgent
    exit 1
fi

echo "[backup] $(date -Is) rsync $SRC/ -> $DEST/"
date -Is > "$ATTEMPT_FILE"   # starts the MIN_INTERVAL_HOURS clock for the next run
rsync_rc=0
# Capture rsync's stderr so we can distinguish benign "vanished" churn (the
# cluster deleting superseded checkpoints mid-transfer) from real read/IO errors.
# stdout (stats2/progress2) still flows straight to the cron log.
err_tmp="$(mktemp "${TMPDIR:-/tmp}/aircc_rsync_err.XXXXXX")"
# APPEND-ONLY, by design. AIRCC deletes the results tree in late August; with --delete the next
# pass after that would faithfully erase the only remaining copy. Nothing here ever removes a file
# from $DEST -- superseded files just accumulate, which is the cheap side of the trade.
#
# Everything is kept now, intermediate checkpoint-N.pth.tar (~1.4GB each) included: this archive is
# the last copy, not a convenience mirror.
#
# That is only affordable because actively-training dirs are skipped -- a running model rewrites its
# checkpoints every few epochs, and re-pulling superseded ones over a ~25MB/s link would never
# converge. A model is picked up in full on the first pass after it leaves 'running'. Failed runs
# get moved out to results/models_failed entirely, so they are archived separately, once.
# AIRCC_BACKUP_SKIP_RUNNING=0 forces a full catch-all sweep (use before the cluster deletion).
running_excludes=()
if [[ "${AIRCC_BACKUP_SKIP_RUNNING:-1}" != "0" ]]; then
    # Read into a variable, NOT `mapfile < <(...)`: a process substitution's exit status is not
    # propagated to mapfile, so a dead DB would look like "no running models" and quietly sync
    # every mid-training dir -- days of link time on checkpoints about to be superseded.
    if ! running_raw="$(python3 - "$AIRCC_DB_PATH" <<'PYEOF'
import sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?immutable=1", uri=True)
for (name,) in con.execute("SELECT model_name FROM jobs WHERE status='running'"):
    print(f"--exclude=/{name}/")
PYEOF
    )"; then
        echo "[backup] ERROR: cannot read job DB $AIRCC_DB_PATH" >&2
        notify "[aircc] backup: cannot read job DB" \
            "$AIRCC_DB_PATH unreadable, so the running-model exclusions could not be built.
Refusing to sync rather than pulling every mid-training checkpoint. Is the sshfs mount up?" \
            "" urgent
        exit 1
    fi
    # Guard the empty case separately: `mapfile <<<""` yields one empty element, which would hand
    # rsync a bare `--exclude=`.
    if [[ -n "$running_raw" ]]; then
        mapfile -t running_excludes <<<"$running_raw"
    fi
    echo "[backup] skipping ${#running_excludes[@]} running model dir(s)"
fi

# -rt (mtime preservation) is load-bearing: aa_sweep/mirror.py only stages a file out of this tree
# when its size AND mtime match the AIRCC source, so dropping -t would silently disable that path.
# protect the script's own log/lock, which live in $DEST rather than under $SRC.
# timeout caps a run that is merely slow or stalled on a live connection; --kill-after escalates to
# SIGKILL if the rsync ignores SIGTERM. A wedged ssh is caught by the stale-lock alert above.
timeout --kill-after=5m "$RSYNC_TIMEOUT" \
    rsync -rt --no-perms --no-owner --no-group --partial \
    --filter='protect /backup.log' --filter='protect /.backup.lock' \
    --filter='protect /.backup.started' --filter='protect /.backup.attempted' \
    "${running_excludes[@]}" --info=stats2,progress2 \
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
