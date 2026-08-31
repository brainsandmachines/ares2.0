#!/bin/bash
# The Botero lane's worker tick. Runs queued AutoAttack sweep units on this machine's RTX 4090,
# strictly one at a time, against models already sitting in the local archives.
#
# Install (Botero crontab -e) -- every 10 minutes. The tick is nearly free when there is nothing to
# do (one sqlite read + one nvidia-smi), and the flock below means a tick landing on a running job
# costs nothing at all. A short interval only decides how fast the lane notices that the GPU came
# free or that the 21:30 cron topped the queue up:
#   */10 * * * * /home/tomer_a/Documents/ares/aa_sweep/scripts/aa_sweep_botero_runner.sh >> /home/tomer_a/Documents/ares/aa_sweep/logs/botero_runner.log 2>&1
#
# The queue itself is filled by `python -m aa_sweep.submit` (the 21:30 cron) and can be inspected or
# edited by hand with `python -m aa_sweep.botero status|enqueue|reset|drop`.
#
# Safe to run by hand. To see only whether the GPU is free:
#   AA_BOTERO_ARGS=--check-gpu aa_sweep/scripts/aa_sweep_botero_runner.sh

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
PYTHON="${AA_SWEEP_PYTHON:-/home/tomer_a/miniconda3/envs/ares/bin/python}"
LOCK_FILE="${AA_BOTERO_LOCK:-$REPO_ROOT/aa_sweep/logs/.botero_runner.lock}"
# Same shape as the daily submitter's hold file: write an ISO timestamp to pause the lane, and it
# clears itself once it expires.
#   date -Is -d 'tomorrow 06:00' > aa_sweep/logs/.botero_hold
HOLD_FILE="${AA_BOTERO_HOLD:-$REPO_ROOT/aa_sweep/logs/.botero_hold}"
EXTRA_ARGS="${AA_BOTERO_ARGS:-}"

mkdir -p -m 0755 "$(dirname "$LOCK_FILE")"

if [[ -f "$HOLD_FILE" ]]; then
    hold_until="$(head -n1 "$HOLD_FILE")"
    if hold_ts="$(date -d "$hold_until" +%s 2>/dev/null)" && [[ "$(date +%s)" -lt "$hold_ts" ]]; then
        echo "[aa_botero] $(date -Is) HOLD: lane paused until $hold_until ($HOLD_FILE)"
        exit 0
    fi
    echo "[aa_botero] $(date -Is) hold expired ($hold_until), resuming and clearing $HOLD_FILE"
    rm -f "$HOLD_FILE"
fi

# THE serialisation point. The lock is held for the entire life of the job -- days, for a full
# 14-cell checkpoint -- so every cron tick in between finds it taken and exits silently. Without
# this two ticks would put two AutoAttack runs on one 24GB card.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    exit 0
fi

cd "$REPO_ROOT" || exit 1

# Streamed through a temp file rather than captured into a variable: a tick holds this lock for the
# whole life of a job -- days for a 14-cell checkpoint -- and `out="$(...)"` would buffer every line
# until then, leaving the cron log looking dead and delaying the failure mail to the very end.
TICK_OUT="$(mktemp)"
trap 'rm -f "$TICK_OUT"' EXIT

# Quiet unless something happened: an idle tick prints one "nothing queued"/"GPU busy" line, and
# every 10 minutes that would bury the log. Those two are dropped as they stream past; everything
# else goes to the log immediately.
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$PYTHON" -u -m aa_sweep.botero_runner ${EXTRA_ARGS} 2>&1 \
    | tee "$TICK_OUT" \
    | grep --line-buffered -vE '^\[aa_botero\].*(nothing queued|GPU busy)'
rc="${PIPESTATUS[0]}"
out="$(cat "$TICK_OUT")"

if [[ "$rc" -ne 0 && -z "$EXTRA_ARGS" ]]; then
    echo "[aa_botero] $(date -Is) ERROR: runner exited rc=$rc" >&2
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$PYTHON" - "$out" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

emailer = make_emailer(source="aa_sweep")
if emailer is None:
    print("[notify] no emailer configured; would send: [aa_sweep] botero runner failed", file=sys.stderr)
else:
    emailer("[aa_sweep] botero runner failed", sys.argv[1], dedup_key="aa_sweep-botero-runner-failed")
PYEOF
fi

exit "$rc"
