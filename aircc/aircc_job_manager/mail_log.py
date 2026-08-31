#!/usr/bin/env python3
"""Read back the alert mail archive that ``notify.archive_mail`` writes.

Every alert any notifier in the repo sends -- spooled, mailed, or urgent -- is
appended to ``logs/mail/YYYY-MM.jsonl``. This is the reader for it:

    python -m aircc.aircc_job_manager.mail_log                    # last 7 days, one line each
    python -m aircc.aircc_job_manager.mail_log --since 30d --source aircc.backup
    python -m aircc.aircc_job_manager.mail_log --grep rsync --full
    python -m aircc.aircc_job_manager.mail_log --id 4f2a1c9b0e77  # one alert, in full

The archive is plain JSONL by design, so ``jq`` over the same files works just as
well; this exists to save you writing the time filter every time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

from aircc.aircc_job_manager.notify import DEFAULT_ARCHIVE_DIR, archive_path

_SINCE_UNITS = {"d": "days", "h": "hours", "w": "weeks"}


def parse_since(raw: str) -> datetime:
    """``7d`` / ``36h`` / ``2w``, or an ISO date like ``2026-08-01``."""
    now = datetime.now().astimezone()
    match = re.fullmatch(r"(\d+)([dhw])", raw.strip().lower())
    if match:
        return now - timedelta(**{_SINCE_UNITS[match.group(2)]: int(match.group(1))})
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a duration (7d/36h/2w) or ISO date: {raw!r}") from None
    return parsed if parsed.tzinfo else parsed.astimezone()


def archive_dir() -> Path:
    """The directory the archive currently writes to (honours ARES_MAIL_ARCHIVE)."""
    path = archive_path()
    return path.parent if path is not None else DEFAULT_ARCHIVE_DIR


def read_records(directory: Path) -> Iterator[dict]:
    """Every archived record, oldest file first. Unparseable lines are skipped."""
    for path in sorted(directory.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"[mail_log] cannot read {path}: {exc}", file=sys.stderr)
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[mail_log] skipping unparseable line in {path}", file=sys.stderr)


def matches(record: dict, *, since: Optional[datetime], source: Optional[str],
            pattern: Optional[re.Pattern], urgent_only: bool,
            msg_id: Optional[str]) -> bool:
    if msg_id:
        return record.get("msg_id", "").startswith(msg_id)
    if urgent_only and not record.get("urgent"):
        return False
    if source and not record.get("source", "").startswith(source):
        return False
    if since:
        try:
            ts = datetime.fromisoformat(record["ts"])
        except (KeyError, ValueError):
            return False        # undatable record, and we were asked for a window
        if ts < since:
            return False
    if pattern and not pattern.search(f"{record.get('subject', '')}\n{record.get('body', '')}"):
        return False
    return True


def format_one_line(record: dict) -> str:
    flags = "".join((
        "!" if record.get("urgent") else " ",
        "~" if record.get("routed") == "spooled" else " ",
        "x" if record.get("send_error") else " ",
    ))
    return (f"{record.get('ts', '?'):<25} {flags} {record.get('source', '?'):<22} "
            f"{record.get('subject', '')}")


def format_full(record: dict) -> str:
    rule = "=" * 72
    head = [rule, f"{record.get('subject', '')}",
            f"  {record.get('source', '?')} · {record.get('ts', '?')} · "
            f"{record.get('routed', '?')}"
            f"{' · URGENT' if record.get('urgent') else ''}"
            f"{' · id ' + record['msg_id'] if record.get('msg_id') else ''}"]
    if record.get("dedup_key"):
        head.append(f"  dedup_key: {record['dedup_key']}")
    if record.get("send_error"):
        head.append(f"  SEND FAILED: {record['send_error']}")
    head.extend(["-" * 72, record.get("body", "").rstrip(), ""])
    return "\n".join(head)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aircc.aircc_job_manager.mail_log",
        description="Read back the archived notifier alerts.")
    parser.add_argument("--since", default="7d",
                        help="window: 7d / 36h / 2w, an ISO date, or 'all' (default: 7d)")
    parser.add_argument("--source", help="only this source (prefix match, e.g. aircc.)")
    parser.add_argument("--grep", help="regex over subject and body (case-insensitive)")
    parser.add_argument("--urgent", action="store_true", help="only urgent alerts")
    parser.add_argument("--id", dest="msg_id",
                        help="one record by msg_id prefix (implies --full, ignores filters)")
    parser.add_argument("--full", action="store_true", help="print bodies, not one line each")
    parser.add_argument("--json", action="store_true", help="print the raw JSONL records")
    parser.add_argument("--dir", type=Path, default=None, help="archive directory to read")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    directory = args.dir or archive_dir()
    if not directory.is_dir():
        print(f"No mail archive at {directory} (nothing archived yet).", file=sys.stderr)
        return 1

    since = None if args.msg_id or args.since.lower() == "all" else parse_since(args.since)
    pattern = re.compile(args.grep, re.IGNORECASE) if args.grep else None

    hits = [r for r in read_records(directory)
            if matches(r, since=since, source=args.source, pattern=pattern,
                       urgent_only=args.urgent, msg_id=args.msg_id)]
    if not hits:
        print("No archived alerts match.", file=sys.stderr)
        return 1

    for record in hits:
        if args.json:
            print(json.dumps(record, ensure_ascii=False))
        elif args.full or args.msg_id:
            print(format_full(record))
        else:
            print(format_one_line(record))

    if not (args.full or args.json or args.msg_id):
        print(f"\n{len(hits)} alert(s). Flags: ! urgent, ~ spooled, x send failed. "
              f"Bodies: --full, or --id <msg_id>.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
