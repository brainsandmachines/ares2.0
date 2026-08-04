"""Email alerts for the AIRCC job manager (status/backup checks).

Self-contained: reads ``AIRCC_ALERT_EMAIL`` / ``AIRCC_SMTP_*`` from the real
environment or ``aircc_job_manager/.env``, and prefers SMTP (no local MTA
needed) over the local ``mail``/``mailx`` command when configured.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("aircc.notify")

_ENV_PATH = Path(__file__).resolve().parent / ".env"

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


def make_emailer() -> Optional[Callable[[str, str], None]]:
    """Return an emailer, or None if no transport is configured.

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

        return _clamped(_send_smtp)

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

    return _clamped(_send)
