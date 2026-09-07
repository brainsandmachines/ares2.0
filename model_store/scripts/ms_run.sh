#!/bin/bash
# Generic runner for a model_store pass. One script rather than one per pass:
# every pass needs the same five things (a lock, an ISO-stamped log, mount guards,
# a failure email, and a non-zero exit that tmux leaves on screen), and the passes
# differ only in their argv.
#
# Usage:
#   model_store/scripts/ms_run.sh <pass> [-- <extra args passed to the module>]
#
# Passes:
#   dupes        Step 1  QNAP duplicate report                      (read-only)
#   dupes-full           same, but sha256 every pair over CIFS      (read-only, ~4h)
#   conflicts    Step 2  data4t merge-decision list, sha256 all     (read-only)
#   conflicts-fast       same, mtime pre-filter only                (read-only)
#   promotions   Step 5  recover which kind /mnt/data promoted      (read-only)
#   build        Step 3  plan the hardlink tree                     (dry run)
#   build-apply  Step 3  build /mnt/data4t/models                   (WRITES)
#   backfill     Step 4  plan the QNAP backfill                     (dry run)
#   backfill-apply       pull what data4t is missing                (WRITES)
#                        -- --roots qnap-slurm   limits the source archive; this is
#                        what the Monday ms_weekly_sync.sh cron passes
#   zoo          Step 6  plan the symlink zoo                       (dry run)
#   zoo-apply    Step 6  build models_for_experiments               (WRITES)
#   stage-data4t Step 7  plan the pending_deletion staging          (dry run)
#   stage-data4t-apply   mv the leftovers into pending_deletion     (MOVES)
#   stage-data   Step 8  plan the /mnt/data staging                 (dry run)
#   stage-data-apply     mv verified duplicates into pending_deletion (MOVES)
#   census               counts by arch/protocol/source             (read-only)
#   legacy               list everything routed to models/_legacy/  (read-only)
#
# Every -apply pass has a dry-run twin with the same name minus the suffix. No pass
# ever deletes: the stage-* passes rename into pending_deletion/<date>/ and leave
# the erase to you.
#
# Why `dupes` does NOT hash everything: both its sides live on the CIFS share, so
# --hash-all there means ~756 GiB of reads at ~55 MB/s (~4h) to produce a report
# whose recommended action is "change nothing". The mtime pre-filter still hashes
# every pair whose mtimes disagree -- i.e. every candidate conflict -- and the
# report labels each verdict's evidence, so an assumed-identical pair is visible
# as such. Use `dupes-full` if you ever want the exhaustive version.
#
# Each pass runs in its own tmux session by convention:
#   tmux new -s ms_dupes 'model_store/scripts/ms_run.sh dupes; read'
#
# Logs land in slurm_job_manager/logs/reorg/ -- local disk, gitignored, and the
# same dir the existing backup lock/stamp files use. Never on /mnt/botero: flock
# over CIFS is not dependable and a redirect into an unmounted share would defeat
# the mount guard.
#
# Exit codes: 0 = ok, 75 = nothing to do (another run holds the lock), else failure.

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
LOG_DIR="${MS_LOG_DIR:-$REPO_ROOT/slurm_job_manager/logs/reorg}"
# The `ares` conda env, not the base interpreter: `build_experiments` scores
# checkpoints through aircc/aircc_job_manager/best_checkpoint.py, which needs
# pandas, and base python3 here does not have it. Falling back to python3 would
# leave the AutoAttack-sweep rule quietly unavailable, so prefer the env and let
# the module itself fail loudly if the dep is still missing.
MS_PYTHON="${MS_PYTHON:-/home/tomer_a/miniconda3/envs/ares/bin/python}"
[[ -x "$MS_PYTHON" ]] || MS_PYTHON="python3"
QNAP_ROOT="${MS_QNAP_ROOT:-/mnt/botero}"
DATA4T_ROOT="${MS_DATA4T_ROOT:-/mnt/data4t}"
EXIT_SKIPPED=75

PASS="${1:-}"
shift || true
[[ "${1:-}" == "--" ]] && shift

if [[ -z "$PASS" ]]; then
    sed -n '10,25p' "$0" | sed 's/^# \?//' >&2
    echo "usage: $0 <pass> [-- args...]" >&2
    exit 2
fi

mkdir -p -m 0755 "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PASS}.log"
LOCK_FILE="$LOG_DIR/.${PASS}.lock"
STAMP_FILE="$LOG_DIR/.${PASS}.started"

log() { echo "[ms:$PASS] $(date -Is) $*"; }

notify() {
    local subject="$1" body="$2" dedup="${3:-}" urgent="${4:-}"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python3 - "$subject" "$body" "$dedup" "$urgent" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body, dedup, urgent = sys.argv[1:5]
emailer = make_emailer(source="model_store")
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body, dedup_key=dedup or None, urgent=bool(urgent))
PYEOF
}

# --- mount guards --------------------------------------------------------
# Only the passes that actually read a mount check it. A CIFS share that is not
# mounted turns $QNAP_ROOT into an empty dir on /, which would make a duplicate
# report cheerfully conclude "no duplicates" -- fail closed instead.
require_qnap() {
    if ! mountpoint -q "$QNAP_ROOT"; then
        log "ERROR: $QNAP_ROOT is not a mountpoint -- refusing to run"
        notify "[model_store] $PASS aborted: $QNAP_ROOT not mounted" \
"$QNAP_ROOT is not a mountpoint, so it is an empty directory on the root filesystem.
A report generated against it would be silently wrong, so this run refused.

Remount the QNAP share and run again." "ms-$PASS-not-mounted" urgent
        exit 1
    fi
    # A mountpoint test alone passes on a stale or entirely different share, so
    # also check a sentinel that only exists on this one.
    if [[ ! -d "$QNAP_ROOT/aircc_archive" ]]; then
        log "ERROR: sentinel $QNAP_ROOT/aircc_archive missing -- refusing to run"
        notify "[model_store] $PASS aborted: sentinel missing" \
"$QNAP_ROOT is mounted but $QNAP_ROOT/aircc_archive does not exist.
That usually means a different share is mounted there." "ms-$PASS-no-sentinel" urgent
        exit 1
    fi
}

require_data4t() {
    if [[ ! -d "$DATA4T_ROOT/slurm_archive" || ! -d "$DATA4T_ROOT/aircc_archive" ]]; then
        log "ERROR: $DATA4T_ROOT does not hold both archives -- refusing to run"
        exit 1
    fi
}

# --- single-instance lock ------------------------------------------------
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    held="$(cat "$STAMP_FILE" 2>/dev/null || echo unknown)"
    log "SKIP: another $PASS run holds $LOCK_FILE (since $held)"
    exit "$EXIT_SKIPPED"
fi
date -Is > "$STAMP_FILE"
trap 'rm -f "$STAMP_FILE"' EXIT

# --- dispatch ------------------------------------------------------------
declare -a CMD
case "$PASS" in
    dupes)
        require_qnap
        CMD=("$MS_PYTHON" -m model_store.dedupe_report --scope qnap)
        ;;
    dupes-full)
        require_qnap
        CMD=("$MS_PYTHON" -m model_store.dedupe_report --scope qnap --hash-all)
        ;;
    conflicts)
        require_data4t
        CMD=("$MS_PYTHON" -m model_store.dedupe_report --scope data4t --hash-all)
        ;;
    conflicts-fast)
        require_data4t
        CMD=("$MS_PYTHON" -m model_store.dedupe_report --scope data4t)
        ;;
    promotions)
        require_data4t
        if [[ ! -d "${MS_DATA_ROOT:-/mnt/data}/robustness_models" ]]; then
            log "ERROR: ${MS_DATA_ROOT:-/mnt/data}/robustness_models missing"
            exit 1
        fi
        CMD=("$MS_PYTHON" -m model_store.promotions --report)
        ;;
    build|build-apply)
        require_data4t
        CMD=("$MS_PYTHON" -m model_store.build_models)
        [[ "$PASS" == "build-apply" ]] && CMD+=(--apply)
        ;;
    backfill|backfill-apply)
        require_qnap
        require_data4t
        CMD=("$MS_PYTHON" -m model_store.backfill)
        [[ "$PASS" == "backfill-apply" ]] && CMD+=(--apply)
        ;;
    zoo|zoo-apply)
        require_data4t
        CMD=("$MS_PYTHON" -m model_store.build_experiments)
        [[ "$PASS" == "zoo-apply" ]] && CMD+=(--apply)
        ;;
    stage-data4t|stage-data4t-apply)
        require_data4t
        CMD=("$MS_PYTHON" -m model_store.stage --scope data4t)
        [[ "$PASS" == "stage-data4t-apply" ]] && CMD+=(--apply)
        ;;
    stage-data|stage-data-apply)
        CMD=("$MS_PYTHON" -m model_store.stage --scope data)
        [[ "$PASS" == "stage-data-apply" ]] && CMD+=(--apply)
        ;;
    census)
        CMD=("$MS_PYTHON" -m model_store.census --summary)
        ;;
    legacy)
        CMD=("$MS_PYTHON" -m model_store.census --report-unparsed)
        ;;
    *)
        log "ERROR: unknown pass '$PASS'"
        exit 2
        ;;
esac

{
    log "start: ${CMD[*]} $*"
    started=$(date +%s)
    cd "$REPO_ROOT" || exit 1
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 "${CMD[@]}" "$@"
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    if [[ "$rc" -eq 0 ]]; then
        log "done rc=0 after $((elapsed / 60))m $((elapsed % 60))s"
    else
        log "ERROR: rc=$rc after $((elapsed / 60))m $((elapsed % 60))s"
        notify "[model_store] $PASS failed rc=$rc" \
"${CMD[*]} exited $rc after $((elapsed / 60))m.

Tail of $LOG_FILE:
$(tail -30 "$LOG_FILE" 2>/dev/null)" "" urgent
    fi
    exit "$rc"
} 2>&1 | tee -a "$LOG_FILE"

exit "${PIPESTATUS[0]}"
