#!/usr/bin/env python3
"""Hourly watch: email once when the sjm DB drains -- no row pending or running.

    python -m slurm_job_manager.watch_drained            # from cron
    python -m slurm_job_manager.watch_drained --dry-run  # print the mail; send and save nothing

Runs on Botero and reads the live DB over the sshfs mount (immutable: no lock taken over NFS).
"Drained" means every row is ``finished`` or ``failed``. The mail lists the per-status counts and
every failed model with the tail of its ``last_error``. Sent ``urgent`` so it is mailed on the spot
rather than held for the 07:30 digest.

It mails once per drain: state lives in ``logs/watch_drained.json``, and if rows go back to
pending/running later (new seed, a manual requeue) the watch re-arms and mails on the next drain.
A missing DB (mount dropped) is logged and never counts as drained.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

SJM_DB = Path.home() / "slurm_mount/projects/ares/slurm_job_manager/jobs.sqlite"
LOG_DIR = Path(__file__).resolve().parent / "logs"
STATE_PATH = LOG_DIR / "watch_drained.json"
ACTIVE = ("pending", "running")


def log(msg: str) -> None:
    print(f"[watch_drained] {datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def read_rows() -> list[dict]:
    conn = sqlite3.connect(f"file:{SJM_DB}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT model_name, status, current_epoch, total_epochs, last_error FROM jobs "
            "ORDER BY status, model_name")]
    finally:
        conn.close()


def mail(subject: str, body: str, dry_run: bool) -> None:
    if dry_run:
        print(f"--- DRY-RUN mail ---\nSubject: {subject}\n\n{body}\n--------------------")
        return
    from aircc.aircc_job_manager.notify import make_emailer

    emailer = make_emailer(source="sjm.watch_drained")
    if emailer is None:
        raise RuntimeError("no emailer configured")
    emailer(subject, body, urgent=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="slurm_job_manager.watch_drained")
    ap.add_argument("--dry-run", action="store_true", help="print the mail; send and save nothing")
    args = ap.parse_args(argv)

    state: dict = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"notified": False}

    if not SJM_DB.exists():
        log(f"ERROR {SJM_DB} not found (is ~/slurm_mount mounted?)")
        return 1
    try:
        rows = read_rows()
    except sqlite3.Error as exc:
        log(f"ERROR reading {SJM_DB}: {exc}")
        return 1

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    active = [r for r in rows if r["status"] in ACTIVE]

    if active or not rows:
        if state.get("notified"):
            log("rows active again; re-armed for the next drain")
        state = {"notified": False}
        log(f"not drained: {summary or 'empty DB'}")
    elif state.get("notified"):
        log(f"drained, already mailed at {state.get('mailed_at')}: {summary}")
        return 0
    else:
        failed = [r for r in rows if r["status"] == "failed"]
        lines = [f"The sjm DB has no pending or running rows: {summary}", ""]
        if failed:
            lines.append(f"failed ({len(failed)}):")
            for r in failed:
                lines.append(f"- {r['model_name']}  epoch {r['current_epoch']}/{r['total_epochs']}")
                if r.get("last_error"):
                    tail = str(r["last_error"]).strip().splitlines()[-3:]
                    lines += [f"    {ln.strip()[:200]}" for ln in tail]
        else:
            lines.append("no failed rows.")
        try:
            mail(f"[sjm] DB drained: {summary}", "\n".join(lines), args.dry_run)
        except Exception as exc:
            log(f"MAIL FAILED {exc}")
            return 1
        log(f"drained -> mailed: {summary}" + (" (dry-run)" if args.dry_run else ""))
        if args.dry_run:
            return 0
        state = {"notified": True, "mailed_at": datetime.now().isoformat(timespec="seconds"),
                 "summary": summary}

    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
