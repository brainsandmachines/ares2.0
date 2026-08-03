#!/bin/bash
# Daily AutoAttack full-sweep completion. Reads both job DBs read-only over the sshfs mounts,
# finds finished models whose linf/l2/l1 x eps 1,2,4,6,8 grid is incomplete on best/last/advbest,
# stages any missing AIRCC checkpoints onto the BGU cluster, and submits one sbatch per
# (model, checkpoint kind) on the `main` partition. Quiet unless something breaks.
#
# Install (Botero crontab -e) -- 19:30 is 16.5h after the 03:00 AIRCC backup, so a long backup has
# finished and the local mirror it fills is usable as the staging source (see aa_sweep/mirror.py):
#   30 19 * * * /home/tomer_a/Documents/ares/aa_sweep/scripts/aa_sweep_daily.sh >> /home/tomer_a/Documents/ares/aa_sweep/logs/aa_sweep.log 2>&1
#
# Safe to run by hand; add --dry-run via AA_SWEEP_ARGS to see the plan without submitting:
#   AA_SWEEP_ARGS=--dry-run aa_sweep/scripts/aa_sweep_daily.sh

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
PYTHON="${AA_SWEEP_PYTHON:-/home/tomer_a/miniconda3/envs/ares/bin/python}"
LOCK_FILE="${AA_SWEEP_LOCK:-$REPO_ROOT/aa_sweep/logs/.aa_sweep.lock}"
EXTRA_ARGS="${AA_SWEEP_ARGS:-}"

mkdir -p -m 0755 "$(dirname "$LOCK_FILE")"

# A sweep run can take minutes (staging GBs); never let two overlap.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[aa_sweep] $(date -Is) SKIP: another run already holds $LOCK_FILE" >&2
    exit 0
fi

notify() {
    local subject="$1" body="$2"
    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$PYTHON" - "$subject" "$body" <<'PYEOF'
import sys
from aircc.aircc_job_manager.notify import make_emailer

subject, body = sys.argv[1], sys.argv[2]
emailer = make_emailer()
if emailer is None:
    print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
else:
    emailer(subject, body)
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
        notify "[aa_sweep] daily run failed rc=$rc" "$out"
    fi
fi

exit "$rc"
