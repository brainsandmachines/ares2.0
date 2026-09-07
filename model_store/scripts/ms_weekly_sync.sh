#!/bin/bash
# ROUTE 2 of the weekly model sync: /mnt/botero/slurm_archive -> /mnt/data4t/models
# -> the models_for_experiments symlinks.
#
#     slurm results/models  --(Sun 09:00, slurm_job_manager/scripts/backup_slurm_models.sh)-->
#     /mnt/botero/slurm_archive  --(Mon 09:00, THIS SCRIPT)-->
#     /mnt/data4t/models  <--  /mnt/data4t/models_for_experiments  <--  epsilon_bounded_contstim
#
# The two routes are independent on purpose. Route 1 pulls ~0.9TB over ssh and must not be
# reported as failed because an index rebuild afterwards failed; route 2 must be re-runnable
# on its own without re-pulling any of that. They share only the archive between them, and
# the lock in step 0.
#
# Why this is a wrapper rather than a script that does the work: ms_run.sh already carries a
# per-pass flock, an ISO-stamped log, the QNAP mount + sentinel guards, the `ares` interpreter
# (build_experiments needs pandas), and a failure email. Reimplementing any of that here would
# be a second copy to keep in sync. Same shape as ms_finish.sh.
#
# Install (Botero crontab -e):
#   0 9 * * 1 /home/tomer_a/Documents/ares/model_store/scripts/ms_weekly_sync.sh >> /home/tomer_a/Documents/ares/slurm_job_manager/logs/reorg/weekly_sync.log 2>&1
#
#   ms_weekly_sync.sh --dry-run     # plan both passes, write nothing
#
# Exit codes: 0 = ok (or nothing to do), 75 = skipped because route 1 is still pulling,
# anything else = a real failure (ms_run.sh has already emailed about it).

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
LOG_DIR="${MS_LOG_DIR:-$REPO_ROOT/slurm_job_manager/logs/reorg}"
RUN="$REPO_ROOT/model_store/scripts/ms_run.sh"
EXPERIMENTS_ROOT="${MS_EXPERIMENTS_ROOT:-/mnt/data4t/models_for_experiments}"
# Route 1's lock, on local disk. Named here rather than sourced because backup_slurm_models.sh
# is not sourceable -- it runs its own preflight at import time.
BACKUP_LOCK="${SJM_BACKUP_LOCK:-$REPO_ROOT/slurm_job_manager/logs/.backup.lock}"
MS_PYTHON="${MS_PYTHON:-/home/tomer_a/miniconda3/envs/ares/bin/python}"
[[ -x "$MS_PYTHON" ]] || MS_PYTHON="python3"
# Route 2 pulls from the Slurm archive only. The AIRCC allocation is over and its archive is
# static; `ms_run.sh backfill-apply` with no --roots still covers it by hand when wanted.
ROOTS="${MS_WEEKLY_ROOTS:-qnap-slurm}"
EXIT_SKIPPED=75

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p -m 0755 "$LOG_DIR"
log() { echo "[weekly-sync] $(date -Is) $*"; }

cd "$REPO_ROOT" || { log "ERROR: cannot cd to $REPO_ROOT"; exit 1; }

# --- 0. do not index a half-finished pull --------------------------------
# Route 1 starts Sunday 09:00 and takes 4-5h, so on a normal week this lock is long free by
# Monday. On a slow or hand-restarted week it is not, and backfilling from an archive that
# is still being written would publish a model whose checkpoint is mid-flight -- the exact
# property route 1's old `overall_rc` gate used to give. Take the lock only to test it, and
# drop it immediately: holding it through a multi-hour backfill would block route 1 itself.
if [[ -e "$BACKUP_LOCK" ]]; then
    if ! flock -n "$BACKUP_LOCK" true; then
        log "SKIP: route 1 (backup_slurm_models.sh) still holds $BACKUP_LOCK"
        exit "$EXIT_SKIPPED"
    fi
fi

# --- 1. route 2's rsync: QNAP -> /mnt/data4t/models ------------------------
# Update-only: backfill pulls a file when it is missing, when a checkpoint's QNAP copy is at a
# HIGHER EPOCH, or when metadata is genuinely newer -- and passes rsync --update on every leg
# that is not carrying an epoch decision. Intermediate checkpoint-N/tmp are never pulled.
if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY RUN: planning the backfill (--roots $ROOTS)"
    "$RUN" backfill -- --roots $ROOTS
else
    log "route 2: backfill $ROOTS -> /mnt/data4t/models"
    "$RUN" backfill-apply -- --roots $ROOTS
fi
rc=$?
if [[ "$rc" -eq "$EXIT_SKIPPED" ]]; then
    log "SKIP: another backfill run holds the pass lock"
    exit "$EXIT_SKIPPED"
elif [[ "$rc" -ne 0 ]]; then
    log "ERROR: backfill failed rc=$rc -- not rebuilding the zoo off a partial pull"
    exit "$rc"
fi

# --- 2. the zoo's blessing source must actually be readable ----------------
# build_experiments only publishes a model the job DB recorded a best_checkpoint for, and
# census._read_db returns [] for a DB path that does not exist -- which is exactly how
# ~/slurm_mount looks when the sshfs has dropped. That turns every Slurm model into
# "not-db-blessed", and step 3's --delete then prunes them out of the live tree. Fail closed.
SJM_DB="${MS_SJM_DB:-$HOME/slurm_mount/projects/ares/slurm_job_manager/jobs.sqlite}"
if ! "$MS_PYTHON" - "$SJM_DB" <<'PYCHECK'
import sqlite3, sys
from pathlib import Path
db = Path(sys.argv[1])
if not db.exists():
    sys.exit(f"[weekly-sync] ERROR: {db} does not exist -- is ~/slurm_mount up? "
             f"(mount | grep slurm_mount; the mount drops silently)")
try:
    con = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    n = con.execute(
        "SELECT count(*) FROM jobs WHERE best_checkpoint IS NOT NULL").fetchone()[0]
except sqlite3.Error as exc:
    sys.exit(f"[weekly-sync] ERROR: cannot read {db}: {exc}")
if n == 0:
    sys.exit(f"[weekly-sync] ERROR: {db} has 0 rows with a best_checkpoint -- "
             f"refusing to rebuild a zoo that would prune every Slurm model")
print(f"[weekly-sync] sjm DB ok: {n} blessed rows")
PYCHECK
then
    exit 1
fi

# A second floor, in case some other census input comes back thin: refuse a plan that would
# drop more than ~20% of what is already published. A genuine large prune stays possible by
# hand -- `ms_run.sh zoo-apply` on its own passes no --min-entries.
live_entries="$(find "$EXPERIMENTS_ROOT" -name '*.pth.tar' 2>/dev/null | wc -l)"
min_entries=$(( live_entries * 80 / 100 ))
log "live zoo has $live_entries entries; floor for this rebuild is $min_entries"

# --- 3. rebuild the experiment symlinks ------------------------------------
# The links are relative and resolve into /mnt/data4t/models, never the QNAP: experiments on
# this machine read local copies. This is the one --delete in model_store, and it prunes stale
# symlinks only -- it cannot reach a checkpoint.
if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY RUN: planning the experiment zoo"
    "$RUN" zoo -- --min-entries "$min_entries"
else
    log "rebuilding $EXPERIMENTS_ROOT"
    "$RUN" zoo-apply -- --min-entries "$min_entries"
fi
rc=$?
if [[ "$rc" -eq "$EXIT_SKIPPED" ]]; then
    log "SKIP: another zoo run holds the pass lock"
    exit "$EXIT_SKIPPED"
elif [[ "$rc" -ne 0 ]]; then
    log "ERROR: zoo rebuild failed rc=$rc"
    exit "$rc"
fi

[[ "$DRY_RUN" -eq 1 ]] && { log "DRY RUN done -- nothing written"; exit 0; }

# --- 4. verify what experiments will actually load -------------------------
# A dangling link is the one failure mode that is silent at build time and loud only when a
# run tries to load the checkpoint, so check it here rather than leaving it to discovery.
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$MS_PYTHON" -m model_store.build_experiments --check
rc=$?
if [[ "$rc" -ne 0 ]]; then
    log "ERROR: zoo --check reported problems rc=$rc"
    exit "$rc"
fi

dangling="$(find "$EXPERIMENTS_ROOT" -xtype l 2>/dev/null | wc -l)"
offsite="$(find "$EXPERIMENTS_ROOT" -lname '*/mnt/botero/*' 2>/dev/null | wc -l)"
entries="$(find "$EXPERIMENTS_ROOT" -name '*.pth.tar' 2>/dev/null | wc -l)"
if [[ "$dangling" -ne 0 ]]; then
    log "ERROR: $dangling dangling symlink(s) in $EXPERIMENTS_ROOT"
    exit 1
fi
if [[ "$offsite" -ne 0 ]]; then
    log "ERROR: $offsite symlink(s) point at the QNAP instead of local /mnt/data4t"
    exit 1
fi
log "done: $entries entries, 0 dangling, 0 pointing off /mnt/data4t"
