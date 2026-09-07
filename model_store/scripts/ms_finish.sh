#!/bin/bash
# Unattended tail of the reorg: waits for the QNAP backfill, then runs Steps 6, 6b,
# 7 and 8 in order, stopping at the first failure.
#
# Nothing here deletes. Steps 7 and 8 *rename* files into pending_deletion/<date>/,
# which is why they free no space by themselves -- that is the point: the user
# erases those directories by hand after inspecting them.
#
#   tmux new -d -s ms_finish 'model_store/scripts/ms_finish.sh'
#
# Progress: slurm_job_manager/logs/reorg/finish.log

set -u -o pipefail

REPO_ROOT="${ARES_REPO:-/home/tomer_a/Documents/ares}"
LOG_DIR="$REPO_ROOT/slurm_job_manager/logs/reorg"
LOG="$LOG_DIR/finish.log"
CONTSTIM="${CONTSTIM_REPO:-/home/tomer_a/Documents/epsilon_bounded_contstim}"
RUN="$REPO_ROOT/model_store/scripts/ms_run.sh"
MS_PYTHON="${MS_PYTHON:-/home/tomer_a/miniconda3/envs/ares/bin/python}"
[[ -x "$MS_PYTHON" ]] || MS_PYTHON="python3"

mkdir -p "$LOG_DIR"
log() { echo "[finish] $(date -Is) $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

cd "$REPO_ROOT" || die "cannot cd to $REPO_ROOT"

# --- 0. wait for the two read/write passes already in flight ----------------
wait_for() {  # wait_for <logfile> <label> <max minutes>
    local f="$LOG_DIR/$1" label="$2" max="$3" waited=0
    log "waiting for $label ($f)"
    while true; do
        if [[ -f "$f" ]] && grep -qE 'done rc=0' "$f"; then
            log "$label finished OK"; return 0
        fi
        if [[ -f "$f" ]] && grep -qE 'done rc=[1-9]|ABORT|REFUSING' "$f"; then
            die "$label failed -- see $f"
        fi
        (( waited >= max * 60 )) && die "$label still not done after ${max}m"
        sleep 30; waited=$((waited + 30))
    done
}

wait_for "backfill-apply.log" "Step 4 backfill"   180
wait_for "promotions.log"     "Step 5 promotions" 120

# --- 1. Step 6: the symlink zoo --------------------------------------------
log "Step 6: building models_for_experiments"
"$RUN" zoo-apply >>"$LOG" 2>&1 || die "zoo-apply failed"
"$MS_PYTHON" -m model_store.build_experiments --check >>"$LOG" 2>&1 \
    || die "zoo --check reported problems"
dangling=$(find /mnt/data4t/models_for_experiments -xtype l 2>/dev/null | wc -l)
[[ "$dangling" -eq 0 ]] || die "$dangling dangling symlinks in the zoo"
log "Step 6 done: $(find /mnt/data4t/models_for_experiments -name '*.pth.tar' | wc -l) entries, 0 dangling"

# --- 2. Step 6b: point the contstim repo at the new zoo ---------------------
# Only models_zoo_path moves. models_path still names the flat convnext_small
# files under /mnt/data/robustness_models/madry/l2/init1, which conf/explicit_pairs
# refers to by that flat name and which the curated tree does not reproduce -- so
# that path stays, and stage.KEEP_PREFIXES protects it from Step 8.
CONF="$CONTSTIM/conf/machine/botero.yaml"
if [[ -f "$CONF" ]]; then
    if grep -q '^models_zoo_path: /mnt/data4t/models_for_experiments' "$CONF"; then
        log "Step 6b: $CONF already points at the new zoo"
    else
        cp -p "$CONF" "$CONF.bak-$(date +%Y%m%d%H%M%S)"
        sed -i 's|^models_zoo_path:.*|models_zoo_path: /mnt/data4t/models_for_experiments|' "$CONF"
        grep -q '^models_zoo_path: /mnt/data4t/models_for_experiments' "$CONF" \
            || die "failed to repoint models_zoo_path in $CONF"
        log "Step 6b: repointed models_zoo_path -> /mnt/data4t/models_for_experiments"
    fi
else
    log "Step 6b: SKIP, $CONF not found"
fi

# --- 3. Step 7: stage the data4t leftovers ---------------------------------
log "Step 7: staging /mnt/data4t leftovers (mv only)"
"$RUN" stage-data4t >>"$LOG" 2>&1 || die "stage-data4t dry run refused"
"$RUN" stage-data4t-apply >>"$LOG" 2>&1 || die "stage-data4t-apply failed"

# --- 4. Step 8: stage the /mnt/data duplicates -----------------------------
log "Step 8: staging /mnt/data duplicates (mv only)"
"$RUN" stage-data >>"$LOG" 2>&1 || die "stage-data dry run refused"
"$RUN" stage-data-apply >>"$LOG" 2>&1 || die "stage-data-apply failed"

# --- 5. final verification -------------------------------------------------
log "verifying"
for root in /mnt/data4t/pending_deletion/* /mnt/data/pending_deletion/*; do
    [[ -d "$root" ]] || continue
    "$MS_PYTHON" -m model_store.stage --check-disjoint "$root" >>"$LOG" 2>&1 \
        || die "$root shares an inode with the curated tree"
    log "disjoint OK: $root ($(du -sh "$root" 2>/dev/null | cut -f1))"
done
dangling=$(find /mnt/data4t/models_for_experiments -xtype l 2>/dev/null | wc -l)
[[ "$dangling" -eq 0 ]] || die "$dangling dangling zoo symlinks after staging"
unl=$(find /mnt/data4t/models -type f -links 1 -name '*.pth.tar' | wc -l)
log "zoo dangling: 0 | curated checkpoints with no archive twin: $unl (expected: backfilled ones)"
log "ALL DONE -- nothing was deleted; erase the pending_deletion dirs yourself"
df -h /mnt/data4t /mnt/data | tee -a "$LOG"
