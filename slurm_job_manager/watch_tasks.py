#!/usr/bin/env python3
"""Daily watch: send one email per Slurm array task as each of them ends.

    python -m slurm_job_manager.watch_tasks 21490452_1 21490452_5 21490452_7            # from cron
    python -m slurm_job_manager.watch_tasks 21490452_1 21490452_5 21490452_7 --dry-run  # print only

Runs on Botero. One ``sacct`` over ssh per run. Any state other than RUNNING/PENDING/REQUEUED/
SUSPENDED counts as ended (COMPLETED, FAILED, TIMEOUT, CANCELLED, ...), and each ended task is mailed
exactly once. Its mail names the model the task was training (the sjm DB row it owned) and gives
that row's final status and a tail of the task log. Sent ``urgent`` so it is mailed on the spot
rather than held for the 07:30 digest.

The model is captured while the task is still running, because ``mark_finished`` clears the row's
``slurm_job_id`` and afterwards nothing ties the model back to the task. State lives in
``logs/watch_tasks_<array>.json``. Once every watched task is reported the run says so, and its cron
line can be deleted.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

SSH_HOST = "slurm"
SJM_DB = Path.home() / "slurm_mount/projects/ares/slurm_job_manager/jobs.sqlite"
LOG_DIR = Path(__file__).resolve().parent / "logs"
STILL_GOING = {"RUNNING", "PENDING", "REQUEUED", "SUSPENDED", "CONFIGURING", "COMPLETING", "RESIZING"}
LOG_TAIL_LINES = 20


def log(msg: str) -> None:
    print(f"[watch_tasks] {datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def ssh(cmd: str) -> str:
    proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", SSH_HOST, cmd],
                          capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"ssh `{cmd}` rc={proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def sacct(task_ids: list[str]) -> dict[str, dict]:
    out = ssh(f"sacct -j {','.join(task_ids)} -X -n -P "
              "-o JobID,JobIDRaw,State,Start,End,Elapsed,ExitCode")
    rows = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 7 and parts[0] in task_ids:
            job, raw, state, start, end, elapsed, exit_code = parts
            rows[job] = {"raw": raw, "state": state.split()[0], "start": start, "end": end,
                         "elapsed": elapsed, "exit_code": exit_code}
    return rows


def sjm_row(where: str, param) -> Optional[dict]:
    """One jobs row from the sjm DB over the sshfs mount (immutable: no lock taken over NFS)."""
    if not SJM_DB.exists():
        return None
    conn = sqlite3.connect(f"file:{SJM_DB}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(f"SELECT * FROM jobs WHERE {where}", (param,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def stdout_path(raw_id: str) -> str:
    out = ssh(f"scontrol show job {raw_id} 2>/dev/null | grep -o 'StdOut=[^ ]*' || true")
    return out.strip().partition("=")[2]


def mail(subject: str, body: str, dry_run: bool) -> None:
    if dry_run:
        print(f"--- DRY-RUN mail ---\nSubject: {subject}\n\n{body}\n--------------------")
        return
    from aircc.aircc_job_manager.notify import make_emailer

    emailer = make_emailer(source="sjm.watch_tasks")
    if emailer is None:
        raise RuntimeError("no emailer configured")
    emailer(subject, body, urgent=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="slurm_job_manager.watch_tasks")
    ap.add_argument("tasks", nargs="+", help="array task ids, e.g. 21490452_5")
    ap.add_argument("--dry-run", action="store_true", help="print the mails; send and save nothing")
    args = ap.parse_args(argv)

    array = args.tasks[0].split("_")[0]
    state_path = LOG_DIR / f"watch_tasks_{array}.json"
    state: dict = json.loads(state_path.read_text()) if state_path.exists() else {}

    try:
        acct = sacct(args.tasks)
    except Exception as exc:
        log(f"ERROR {exc}")
        return 1

    for task in args.tasks:
        info = acct.get(task)
        entry = state.setdefault(task, {"notified": False})
        if info is None:
            log(f"{task}: not in sacct")
            continue
        entry["raw"] = info["raw"]
        if info["state"] in STILL_GOING:
            if not entry.get("model"):
                row = sjm_row("slurm_job_id=?", int(info["raw"]))
                entry["model"] = row["model_name"] if row else None
                entry["stdout"] = stdout_path(info["raw"]) or None
            log(f"{task}: {info['state']} {info['elapsed']} model={entry.get('model')}")
            continue
        if entry["notified"]:
            continue

        model = entry.get("model")
        row = sjm_row("model_name=?", model) if model else None
        lines = [
            f"Slurm task {task} (job {info['raw']}) ended: {info['state']} (exit {info['exit_code']})",
            f"start {info['start']}  end {info['end']}  elapsed {info['elapsed']}",
            "",
            f"model: {model or 'unknown (task ended before the watch first saw it)'}",
        ]
        if row:
            lines.append(f"sjm DB: status={row['status']} epoch {row['current_epoch']}/{row['total_epochs']} "
                         f"best_score={row.get('best_score')} requeued={row.get('requeued')}")
            if row.get("last_error"):
                lines += ["", "last_error:", str(row["last_error"])[-1500:]]
        if entry.get("stdout"):
            try:
                tail = ssh(f"tail -n {LOG_TAIL_LINES} {entry['stdout']}")
            except Exception as exc:
                tail = f"(could not read log: {exc})"
            lines += ["", f"log {entry['stdout']} (last {LOG_TAIL_LINES} lines):", tail.rstrip()]
        remaining = [t for t in args.tasks if t != task and not state.get(t, {}).get("notified")]
        lines += ["", f"still watching: {', '.join(remaining) if remaining else 'none -- this was the last one'}"]
        subject = f"[sjm] {task} ended ({info['state']}): {model or 'unknown model'}"
        try:
            mail(subject, "\n".join(lines), args.dry_run)
        except Exception as exc:
            log(f"{task}: MAIL FAILED {exc}")
            continue
        log(f"{task}: {info['state']} -> mailed" + (" (dry-run)" if args.dry_run else ""))
        if not args.dry_run:
            entry["notified"] = True

    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
    if all(state.get(t, {}).get("notified") for t in args.tasks):
        log(f"all {len(args.tasks)} task(s) reported; this cron line can be removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
