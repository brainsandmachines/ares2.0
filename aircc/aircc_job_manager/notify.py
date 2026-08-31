"""Email alerts for the AIRCC job manager (status/backup checks).

Self-contained: reads ``AIRCC_ALERT_EMAIL`` / ``AIRCC_SMTP_*`` from the real
environment or ``aircc_job_manager/.env``, and prefers SMTP (no local MTA
needed) over the local ``mail``/``mailx`` command when configured.

Every notifier in the repo (``daily_monitor``, ``aa_sweep``, the sjm and backup
shell scripts, the catastrophic-overfitting notifier) sends through
``make_emailer`` here, which makes this the one place to batch them. When
``ARES_ALERT_SPOOL`` is set, a normal-urgency alert is appended to a JSONL spool
instead of mailed, and ``digest.py`` mails the whole spool as one message the
next morning. Urgent alerts ignore the spool and go out immediately.

Spooling is off unless ``ARES_ALERT_SPOOL`` is set, so the default behaviour is
byte-for-byte what it was before.

Independently of that, every alert -- spooled, mailed, or urgent -- is appended
to a permanent JSONL archive under ``logs/mail/YYYY-MM.jsonl`` so you can read
back what the notifiers sent you (``python -m aircc.aircc_job_manager.mail_log``).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("aircc.notify")

_ENV_PATH = Path(__file__).resolve().parent / ".env"

# emailer(subject, body, *, dedup_key=None, urgent=False) -> None
Emailer = Callable[..., None]

# Set ARES_ALERT_SPOOL=1 for the default path, or to an explicit path.
SPOOL_ENV = "ARES_ALERT_SPOOL"
DEFAULT_SPOOL_PATH = Path(__file__).resolve().parent / "logs" / "alerts.jsonl"

# Safety net for the one way spooling can lose you an alert: if the digest cron
# stops running, entries would pile up in a file nobody reads. The spool is
# append-only, so its first line is its oldest -- when that is older than this,
# assume the digest is broken and mail immediately instead of spooling.
SPOOL_STALE_HOURS = float(os.environ.get("ARES_ALERT_SPOOL_STALE_HOURS", "36"))

# Every alert is also appended to a permanent JSONL archive, one file per month,
# so past mail can be read back long after the spool that batched it was drained
# (digest.py deletes the spool once the digest is away). Unlike the spool this is
# on by default and covers urgent alerts too -- those never reach the spool at
# all. Set ARES_MAIL_ARCHIVE=0 to switch it off, or to a directory to relocate it.
ARCHIVE_ENV = "ARES_MAIL_ARCHIVE"
DEFAULT_ARCHIVE_DIR = Path(__file__).resolve().parent / "logs" / "mail"

# An alert body can quote whatever it was inspecting, and some of those are
# unbounded: an in-progress `rsync --info=progress2` block is a single
# multi-megabyte "line" of redraws, which SMTP will happily send and no mail
# client will render usefully. Every body goes through _clamp_body first.
MAX_BODY_CHARS = int(os.environ.get("AIRCC_ALERT_MAX_BODY_CHARS", "20000"))
MAX_SUBJECT_CHARS = 200


def _clamp_body(body: str, limit: int = MAX_BODY_CHARS) -> str:
    """Keep an alert readable: split \\r redraws into lines, then head+tail it."""
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    trimmed = len(text) - limit
    return f"{text[:head]}\n\n[... {trimmed:,} characters trimmed ...]\n\n{text[-tail:]}"


def _clamp_subject(subject: str) -> str:
    """One line, bounded -- a newline in a subject is a header injection."""
    return " ".join(subject.split())[:MAX_SUBJECT_CHARS]


def _clamped(send: Callable[[str, str], None]) -> Callable[[str, str], None]:
    def _wrapped(subject: str, body: str) -> None:
        send(_clamp_subject(subject), _clamp_body(body))

    return _wrapped


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE .env parser (no external dependency)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def spool_path() -> Optional[Path]:
    """The JSONL alert spool, or None when spooling is off (the default)."""
    raw = os.environ.get(SPOOL_ENV, "").strip()
    if not raw or raw == "0":
        return None
    return DEFAULT_SPOOL_PATH if raw == "1" else Path(raw).expanduser()


def append_spool(path: Path, *, source: str, subject: str, body: str,
                 dedup_key: Optional[str] = None) -> None:
    """Append one alert record to the JSONL spool.

    flock'd rather than relying on O_APPEND alone: a record can carry a body up
    to MAX_BODY_CHARS (20k), well over PIPE_BUF, so a bare append is not atomic
    against the other cron jobs writing the same spool.
    """
    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "subject": _clamp_subject(subject),
        "body": _clamp_body(body),
    }
    if dedup_key:
        record["dedup_key"] = dedup_key
    _append_jsonl(path, record)


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record as a line, locked against the other cron writers."""
    line = json.dumps(record, ensure_ascii=False) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode("utf-8"))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def archive_path(now: Optional[datetime] = None) -> Optional[Path]:
    """This month's mail-archive file, or None when archiving is switched off.

    One file per month rather than one growing file: at a handful of alerts a
    day nothing needs rotating, and "what did it send in July" stays one `cat`.
    """
    raw = os.environ.get(ARCHIVE_ENV, "").strip()
    if raw == "0":
        return None
    root = Path(raw).expanduser() if raw and raw != "1" else DEFAULT_ARCHIVE_DIR
    return root / f"{(now or datetime.now().astimezone()):%Y-%m}.jsonl"


def archive_mail(*, source: str, subject: str, body: str,
                 dedup_key: Optional[str] = None, urgent: bool = False,
                 routed: str = "mailed", send_error: Optional[str] = None) -> None:
    """Record one outgoing alert in the append-only mail archive.

    Best-effort by construction: a failure to write the archive must never cost
    you the alert it was only meant to remember, so every error here is logged
    and swallowed.
    """
    path = archive_path()
    if path is None:
        return
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    subject = _clamp_subject(subject)
    record = {
        "ts": ts,
        # Stable handle for one alert, so a digest section can be traced back to
        # the entry it collapsed.
        "msg_id": hashlib.sha1(f"{ts}|{source}|{subject}".encode()).hexdigest()[:12],
        "source": source,
        "subject": subject,
        "body": _clamp_body(body),
        "urgent": urgent,
        "routed": routed,
    }
    if dedup_key:
        record["dedup_key"] = dedup_key
    if send_error:
        record["send_error"] = send_error
    try:
        _append_jsonl(path, record)
    except OSError as exc:
        logger.warning("could not archive alert %r to %s: %s", subject, path, exc)


def spool_is_stale(path: Path, max_age_hours: float = SPOOL_STALE_HOURS) -> bool:
    """True when the spool's oldest entry is old enough to mean the digest died.

    Reads only the first line: the spool is append-only, so that is the oldest
    record, and a wedged spool must not cost us a full file read on every alert.
    An unreadable or unparseable spool counts as stale -- mailing immediately is
    the safe direction to fail.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if not first.strip():
        return False
    try:
        ts = datetime.fromisoformat(json.loads(first)["ts"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return True
    return datetime.now().astimezone() - ts > timedelta(hours=max_age_hours)


def make_emailer(*, source: str = "unknown") -> Optional[Emailer]:
    """Return an emailer, or None if no transport is configured.

    The returned callable takes ``(subject, body)`` plus two keyword-only
    options: ``dedup_key`` (the digest collapses a repeat of the same key to one
    line instead of a full section) and ``urgent=True`` (never spooled -- mailed
    the moment it happens). ``source`` labels the entry in the digest.

    Two transports, preferred in order:

    1. **SMTP** (no local MTA needed) when ``AIRCC_SMTP_HOST`` is set. Reads
       ``AIRCC_SMTP_PORT`` (default 587, STARTTLS), ``AIRCC_SMTP_USER`` /
       ``AIRCC_SMTP_PASS`` (e.g. a Gmail address + app password), and optional
       ``AIRCC_SMTP_FROM`` (defaults to the user). Recommended on botero, which
       has no mail daemon.
    2. The local ``mail`` / ``mailx`` command, if one exists.
    """
    for key, val in _load_dotenv(_ENV_PATH).items():
        os.environ.setdefault(key, val)

    alert_email = os.environ.get("AIRCC_ALERT_EMAIL")
    if not alert_email:
        return None

    host = os.environ.get("AIRCC_SMTP_HOST")
    if host:
        port = int(os.environ.get("AIRCC_SMTP_PORT", "587"))
        user = os.environ.get("AIRCC_SMTP_USER")
        password = os.environ.get("AIRCC_SMTP_PASS")
        sender = os.environ.get("AIRCC_SMTP_FROM") or user or alert_email

        def _send_smtp(subject: str, body: str) -> None:
            import smtplib
            from email.message import EmailMessage

            msg = EmailMessage()
            msg["From"] = sender
            msg["To"] = alert_email
            msg["Subject"] = subject
            msg.set_content(body)
            with smtplib.SMTP(host, port, timeout=60) as smtp:
                smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(msg)

        return _spoolable(_clamped(_send_smtp), source)

    mail = shutil.which("mail") or shutil.which("mailx")
    if not mail:
        logger.info("no `mail` command on PATH and no AIRCC_SMTP_HOST; "
                    "skipping email alerts")
        return None

    def _send(subject: str, body: str) -> None:
        subprocess.run(
            [mail, "-s", subject, alert_email],
            input=body,
            text=True,
            check=True,
            timeout=60,
        )

    return _spoolable(_clamped(_send), source)


def _spoolable(send: Callable[[str, str], None], source: str) -> Emailer:
    """Route normal-urgency alerts to the spool when one is configured.

    Whichever way an alert goes, it is also archived (see ``archive_mail``) --
    including one that fails to send, which is recorded with ``send_error`` set.
    """

    def _emit(subject: str, body: str, *, dedup_key: Optional[str] = None,
              urgent: bool = False) -> None:
        spool = spool_path()
        routed = "mailed"
        error: Optional[str] = None
        try:
            if spool is None or urgent:
                send(subject, body)
            elif spool_is_stale(spool):
                logger.warning("alert spool %s is stale (digest not running?); "
                               "mailing %r immediately", spool, subject)
                send(subject, body)
            else:
                append_spool(spool, source=source, subject=subject, body=body,
                             dedup_key=dedup_key)
                routed = "spooled"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            archive_mail(source=source, subject=subject, body=body,
                         dedup_key=dedup_key, urgent=urgent, routed=routed,
                         send_error=error)

    return _emit
