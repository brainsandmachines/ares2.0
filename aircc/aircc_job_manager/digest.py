#!/usr/bin/env python3
"""Mail one morning digest for everything the notifiers spooled overnight.

The five cron notifiers (AIRCC backup + daily_monitor at 03:00, sjm status at
06:00, catastrophic overfitting at 09:00/21:00, aa_sweep at 21:30) all send
through ``notify.make_emailer``. With ``ARES_ALERT_SPOOL`` set, their
normal-urgency alerts land in a JSONL spool instead of the inbox; this module
drains that spool into a single message.

Two properties worth keeping:

* **An empty spool sends nothing.** Silence stays meaningful -- there is no
  "all clear" mail to learn to ignore.
* **Nothing is dropped on failure.** The spool is claimed by an atomic rename
  before it is read, and the claimed file is only deleted once the mail is away.
  A failed send leaves it for the next run.

Urgent alerts (catastrophic overfitting, the backup-is-broken family) bypass all
of this and mail the moment they happen -- see ``notify._spoolable``.

Install (Botero crontab -e), after the 07:00 status-HTML regen:
  30 7 * * * cd /home/tomer_a/Documents/ares && /usr/bin/python3 -m aircc.aircc_job_manager.digest >> /home/tomer_a/Documents/ares/aircc/aircc_job_manager/logs/digest.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from aircc.aircc_job_manager.notify import (
    DEFAULT_SPOOL_PATH,
    MAX_BODY_CHARS,
    Emailer,
    _clamp_body,
    make_emailer,
    spool_path,
)

logger = logging.getLogger("aircc.digest")

DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "logs" / "digest_state.json"

# Ordering only -- everything here is worth reading, this decides what you read
# first. Longest-prefix wins so "[aircc] backup rsync failed" beats "[aircc] ".
TIER_PREFIXES: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, (
        "[aircc] backup rsync failed",
        "[aircc] backup rsync problem",
        "[aircc] backup source not mounted",
        "[aircc] backup lock held",
        "[aircc] DB health problem",
        "[ares] catastrophic overfitting",
    )),
    (2, (
        "[aircc] failure on",
        "[sjm] new failure signature",
        "[sjm] status alert",
        "[aircc] status alert",
        "[aircc] finished rows with no best checkpoint",
        "[aircc] failed rows that may have actually completed",
        "[aircc] best-checkpoint backfill FAILED",
        "[aircc] failed-row reclaim FAILED",
    )),
)
DEFAULT_TIER = 3

# Floor on the per-section body budget. Below this a traceback is too mangled to
# act on, and we would rather overshoot MAX_BODY_CHARS and let the emailer's own
# clamp trim the tail than render twelve unreadable stubs.
MIN_SECTION_CHARS = 1500

_RULE = "=" * 68
_THIN = "-" * 68


# --------------------------------------------------------------------------
# spool claiming
# --------------------------------------------------------------------------

def claim_spool(path: Path) -> list[Path]:
    """Atomically take ownership of the spool, plus any earlier failed claims.

    The rename is what makes concurrent notifiers safe: a writer that opens the
    spool a moment later creates a fresh file and its record waits for tomorrow,
    rather than being read-then-truncated out of existence.
    """
    claimed = sorted(path.parent.glob(f"{path.name}.*.processing"))
    if path.exists() and path.stat().st_size > 0:
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        target = path.with_name(f"{path.name}.{stamp}.processing")
        os.replace(path, target)
        claimed.append(target)
    return claimed


def peek_spool(path: Path) -> list[Path]:
    """The same files claim_spool would take, without taking them (--dry-run)."""
    files = sorted(path.parent.glob(f"{path.name}.*.processing"))
    if path.exists() and path.stat().st_size > 0:
        files.append(path)
    return files


def read_records(paths: Iterable[Path]) -> tuple[list[dict], list[str]]:
    """Parse claimed spool files. Bad lines are reported, never fatal."""
    records: list[dict] = []
    problems: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"cannot read {path}: {exc}")
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{path.name}:{lineno}: unparseable ({exc})")
                continue
            if not isinstance(record, dict) or "subject" not in record:
                problems.append(f"{path.name}:{lineno}: not an alert record")
                continue
            record.setdefault("source", "unknown")
            record.setdefault("body", "")
            record.setdefault("ts", "")
            records.append(record)
    records.sort(key=lambda r: (tier_of(r["subject"]), r.get("ts", "")))
    return records, problems


def tier_of(subject: str) -> int:
    for tier, prefixes in TIER_PREFIXES:
        if any(subject.startswith(prefix) for prefix in prefixes):
            return tier
    return DEFAULT_TIER


# --------------------------------------------------------------------------
# repeat-suppression state
# --------------------------------------------------------------------------

def load_state(path: Path) -> dict[str, dict]:
    """Map dedup_key -> {first_seen, last_seen, subject}. Missing file = empty."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reported = payload.get("reported", {})
        if not isinstance(reported, dict):
            raise ValueError("reported must be an object")
        return {k: v for k, v in reported.items() if isinstance(v, dict)}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read digest state {path}: {exc}") from exc


def save_state(path: Path, reported: dict[str, dict]) -> None:
    """Same durable-write shape the catastrophic notifier uses for its state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reported": reported,
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def partition(records: list[dict], state: dict[str, dict]) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Split into (full sections, still-open one-liners) and the next state.

    A record whose dedup_key is already in state is a condition you have already
    been told about -- it collapses to a line. Keys that stop appearing drop out
    of state, so the same condition recurring after a fix alerts in full again.
    """
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    full: list[dict] = []
    repeats: dict[str, dict] = {}
    next_state: dict[str, dict] = {}

    for record in records:
        key = record.get("dedup_key")
        if not key:
            full.append(record)
            continue
        prior = state.get(key)
        if key in next_state:
            # Same condition twice in one window: already accounted for.
            next_state[key]["last_seen"] = record.get("ts") or now
            continue
        next_state[key] = {
            "first_seen": (prior or {}).get("first_seen") or record.get("ts") or now,
            "last_seen": record.get("ts") or now,
            "subject": record["subject"],
        }
        if prior:
            repeats[key] = {**record, "first_seen": next_state[key]["first_seen"]}
        else:
            full.append(record)

    return full, list(repeats.values()), next_state


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(full: list[dict], repeats: list[dict], problems: list[str]) -> tuple[str, str]:
    total = len(full) + len(repeats)
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    sources = sorted({r["source"] for r in full + repeats})
    subject = (f"[ares] digest: {total} alert(s) from {len(sources)} source(s) ({today})"
               if total else f"[ares] digest: spool problems only ({today})")

    lines: list[str] = []
    if full or repeats:
        oldest = min((r.get("ts") or "" for r in full + repeats), default="")
        lines.append(f"{total} alert(s) spooled since {oldest or 'unknown'}.")
        lines.append("")
        counts: dict[str, int] = {}
        for record in full + repeats:
            counts[record["source"]] = counts.get(record["source"], 0) + 1
        for source in sorted(counts, key=lambda s: (-counts[s], s)):
            lines.append(f"  {counts[source]:>3}  {source}")
        lines.append("")

    if repeats:
        lines.append("Still open -- unchanged since first reported, details omitted:")
        for record in sorted(repeats, key=lambda r: r["subject"]):
            lines.append(f"  - {record['subject']}")
            lines.append(f"      first seen {record.get('first_seen', 'unknown')}")
        lines.append("")

    if problems:
        lines.append("Spool problems (records that could not be read):")
        lines.extend(f"  - {p}" for p in problems)
        lines.append("")

    if full:
        # Split the emailer's body budget across the sections instead of letting
        # the first few spend it all and the rest get clamped into nothing.
        budget = max(MIN_SECTION_CHARS,
                     (MAX_BODY_CHARS - len("\n".join(lines))) // len(full))
        for record in full:
            lines.append(_RULE)
            lines.append(f"## {record['subject']}")
            lines.append(f"   {record['source']} · {record.get('ts', 'unknown')}")
            lines.append(_THIN)
            lines.append(_clamp_body(record.get("body", ""), limit=budget).rstrip())
            lines.append("")

    return subject, "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def run_once(
    *,
    spool: Path,
    state_path: Path,
    dry_run: bool = False,
    keep_claimed: bool = False,
    emailer_factory: Callable[..., Optional[Emailer]] = make_emailer,
) -> int:
    claimed = peek_spool(spool) if dry_run else claim_spool(spool)
    if not claimed:
        logger.info("nothing spooled; no digest sent")
        return 0

    records, problems = read_records(claimed)
    if not records and not problems:
        if not dry_run:
            for path in claimed:
                path.unlink(missing_ok=True)
        logger.info("spool held no records; no digest sent")
        return 0

    try:
        state = load_state(state_path)
    except RuntimeError as exc:
        # An unreadable state file must not swallow the alerts it was only ever
        # meant to summarise: start clean and send everything in full.
        problems.append(f"{exc} (repeat-suppression reset)")
        state = {}

    full, repeats, next_state = partition(records, state)
    subject, body = render(full, repeats, problems)

    if dry_run:
        print(f"Subject: {subject}\n")
        print(body)
        return 0

    # urgent=True: the digest itself must never land back in the spool it drains.
    emailer = emailer_factory(source="ares.digest")
    if emailer is None:
        logger.error("no email transport configured; leaving %d claimed file(s) in place",
                     len(claimed))
        return 1
    try:
        emailer(subject, body, urgent=True)
    except Exception as exc:
        logger.error("digest send failed (%s); leaving %d claimed file(s) for the next run",
                     exc, len(claimed))
        return 1

    try:
        save_state(state_path, next_state)
    except OSError as exc:
        # The mail is already gone, so the alerts are not lost -- but tomorrow
        # will repeat what today collapsed. Worth a nonzero exit, not a retry.
        logger.error("digest sent but state could not be saved: %s", exc)
        if not keep_claimed:
            for path in claimed:
                path.unlink(missing_ok=True)
        return 1

    if not keep_claimed:
        for path in claimed:
            path.unlink(missing_ok=True)
    logger.info("digest sent: %d full, %d still-open, %d problem(s)",
                len(full), len(repeats), len(problems))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mail the spooled ares alert digest.")
    parser.add_argument("--spool", type=Path, default=None,
                        help="spool path (default: ARES_ALERT_SPOOL, else the built-in path)")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the digest instead of mailing it (spool is still claimed)")
    parser.add_argument("--keep-claimed", action="store_true",
                        help="do not delete the claimed spool files after a successful send")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    spool = args.spool or spool_path() or DEFAULT_SPOOL_PATH
    return run_once(spool=spool, state_path=args.state_file,
                    dry_run=args.dry_run, keep_claimed=args.keep_claimed)


if __name__ == "__main__":
    sys.exit(main())
