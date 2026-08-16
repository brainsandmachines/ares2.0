#!/bin/bash
# Daily AutoAttack full-sweep completion. Reads both job DBs read-only over the sshfs mounts,
# finds finished models whose linf/l2/l1 x eps 1,2,4,6,8 grid is incomplete on best/last/advbest,
# stages any missing AIRCC checkpoints onto the BGU cluster, and submits one sbatch per
# (model, checkpoint kind) on the `main` partition. Quiet unless something breaks.
#
# Install (Botero crontab -e) -- 21:30 is 18.5h after the 03:00 AIRCC backup, so even a full backup
# pass has finished and the local mirror it fills is usable as the staging source (see
# aa_sweep/mirror.py). This slot used to be 19:30, but a full pass takes ~16.5h (03:00 -> ~19:35 on
# 2026-08-08) and the cron read the log five minutes too early, correctly rejected the in-progress
# block, and fell back to the slower AIRCC mount for the night:
#   30 21 * * * /home/tomer_a/Documents/ares/aa_sweep/scripts/aa_sweep_daily.sh >> /home/tomer_a/Documents/ares/aa_sweep/logs/aa_sweep.log 2>&1
#
# Safe to run by hand; add --dry-run via AA_SWEEP_ARGS to see the plan without submitting:
#   AA_SWEEP_ARGS=--dry-run aa_sweep/scripts/aa_sweep_daily.sh

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
PYTHON="${AA_SWEEP_PYTHON:-/home/tomer_a/miniconda3/envs/ares/bin/python}"
LOCK_FILE="${AA_SWEEP_LOCK:-$REPO_ROOT/aa_sweep/logs/.aa_sweep.lock}"
# Optional pause: a file holding an ISO timestamp. While now < that timestamp the run is skipped,
# and the file is removed once it expires, so a hold never has to be cleaned up by hand.
#   date -Is -d 'tomorrow 06:00' > aa_sweep/logs/.hold
HOLD_FILE="${AA_SWEEP_HOLD:-$REPO_ROOT/aa_sweep/logs/.hold}"
EXTRA_ARGS="${AA_SWEEP_ARGS:-}"

mkdir -p -m 0755 "$(dirname "$LOCK_FILE")"

if [[ -f "$HOLD_FILE" ]]; then
    hold_until="$(head -n1 "$HOLD_FILE")"
    if hold_ts="$(date -d "$hold_until" +%s 2>/dev/null)" && [[ "$(date +%s)" -lt "$hold_ts" ]]; then
        echo "[aa_sweep] $(date -Is) HOLD: submissions paused until $hold_until ($HOLD_FILE)"
        exit 0
    fi
    echo "[aa_sweep] $(date -Is) hold expired ($hold_until), resuming and clearing $HOLD_FILE"
    rm -f "$HOLD_FILE"
fi

# A sweep run can take minutes (staging GBs); never let two overlap.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[aa_sweep] $(date -Is) SKIP: another run already holds $LOCK_FILE" >&2
    exit 0
fi

notify() {
    local subject="$1" body="$2"
    local dedup="${3:-}" urgent="${4:-}"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$PYTHON" - "$subject" "$body" "$dedup" "$urgent" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body, dedup, urgent = sys.argv[1:5]
emailer = make_emailer(source="aa_sweep")
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body, dedup_key=dedup or None, urgent=bool(urgent))
PYEOF
}

cd "$REPO_ROOT" || exit 1

# submit.py emails on its own for the failures it can describe (mounts down, DB unreadable, rsync
# or sbatch failing). This catches the case where it dies in a way it could not report.
out="$(PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$PYTHON" -m aa_sweep.submit ${EXTRA_ARGS} 2>&1)"
rc=$?
echo "$out"

if [[ "$rc" -ne 0 ]]; then
    echo "[aa_sweep] $(date -Is) ERROR: submit exited rc=$rc" >&2
    if ! grep -q '^\[aa_sweep\].*ABORT' <<<"$out"; then
        notify "[aa_sweep] daily run failed rc=$rc" "$out" "aa_sweep-daily-failed"
    fi
fi

exit "$rc"
