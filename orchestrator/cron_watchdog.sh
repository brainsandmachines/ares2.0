#!/bin/bash
# 20-minute botero cron watchdog.
#
# Responsibilities (backup to the persistent daemon):
#   1. Run one reconciling + top-up pass (run_once) so progress is made even if
#      the daemon is momentarily down.
#   2. Ensure the persistent daemon is alive; (re)start it if not.
#
# Install in crontab (botero):
#   */20 * * * * /home/tomer_a/Documents/ares/orchestrator/cron_watchdog.sh >> \
#       /home/tomer_a/Documents/ares/orchestrator/cron.log 2>&1

set -euo pipefail

REPO="/home/tomer_a/Documents/ares"
PY="/home/tomer_a/miniconda3/envs/ares/bin/python"
LOCKFILE="/tmp/ares_orchestrator.lock"

cd "$REPO"

# 1. One reconciling/top-up pass (safe even if the daemon is also running:
#    DB writes are serialized; tick is idempotent).
"$PY" -m orchestrator.daemon_once || echo "[watchdog] run_once failed"

# 2. Ensure the daemon is alive.
alive=0
if [[ -f "$LOCKFILE" ]]; then
  pid="$(cat "$LOCKFILE" 2>/dev/null || echo 0)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    alive=1
  fi
fi

if [[ "$alive" -eq 0 ]]; then
  echo "[watchdog] daemon not running; starting it"
  rm -f "$LOCKFILE"
  nohup "$PY" -m orchestrator.daemon >> "$REPO/orchestrator/daemon.log" 2>&1 &
  echo "[watchdog] started daemon pid $!"
else
  echo "[watchdog] daemon alive (pid $pid)"
fi
