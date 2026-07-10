"""Pluggable LLM-analyzer and email transports for the failure analyzer.

Both are best-effort and optional: if the diagnosis CLI or a mail transport
isn't available, the analyzer still writes the raw-traceback markdown report and
the loop keeps running. Nothing here is on the critical scheduling path.

The diagnosis CLI is ``codex`` by default (overridable via ``ORCH_LLM_CMD``):
unknown, never-before-seen failure signatures are escalated to ``codex exec``
exactly once (deduped upstream by error hash). Everything up to that escalation
is deterministic.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Callable, Optional

from .config import Config

logger = logging.getLogger("orchestrator.notify")

_LLM_PROMPT = (
    "You are debugging a failed ML training job on a SLURM cluster (ARES, "
    "adversarial robustness, PyTorch/Hydra). Given the traceback below, produce "
    "a concise markdown report with: (1) one-line root cause, (2) the most "
    "likely fix, (3) any config/launcher lines to change. Be specific and short. "
    "Do not modify any files; this is a read-only diagnosis.\n\n"
    "Traceback:\n```\n{tb}\n```\n"
)


def _llm_argv(cmd: str) -> list[str]:
    """Non-interactive argv for the configured diagnosis CLI (prompt via stdin)."""
    base = os.path.basename(cmd)
    if base == "codex":
        # `codex exec` runs non-interactively and reads the prompt from stdin.
        return [cmd, "exec", "--skip-git-repo-check"]
    # claude and compatible CLIs accept a `-p` prompt; we pipe via stdin instead.
    return [cmd, "-p"]


def make_llm_client() -> Optional[Callable[[str], str]]:
    """Return a diagnosis client backed by the ``ORCH_LLM_CMD`` CLI (default codex)."""
    cmd_name = os.environ.get("ORCH_LLM_CMD", "codex")
    cmd = shutil.which(cmd_name)
    if not cmd:
        logger.info("no `%s` CLI on PATH; failure reports will be raw-only", cmd_name)
        return None
    argv = _llm_argv(cmd)

    def _client(traceback_text: str) -> str:
        prompt = _LLM_PROMPT.format(tb=traceback_text[:8000])
        proc = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{cmd_name} CLI failed: {proc.stderr[:500]}")
        return proc.stdout.strip()

    return _client


def make_emailer(cfg: Config) -> Optional[Callable[[str, str], None]]:
    """Return an emailer, or None if no transport is configured.

    Two transports, preferred in order:

    1. **SMTP** (no local MTA needed) when ``ORCH_SMTP_HOST`` is set. Reads
       ``ORCH_SMTP_PORT`` (default 587, STARTTLS), ``ORCH_SMTP_USER`` /
       ``ORCH_SMTP_PASS`` (e.g. a Gmail address + app password), and optional
       ``ORCH_SMTP_FROM`` (defaults to the user). This is the recommended way to
       turn alerts on from botero, which has no mail daemon.
    2. The local ``mail`` / ``mailx`` command, if one exists.
    """
    if not cfg.alert_email:
        return None

    host = os.environ.get("ORCH_SMTP_HOST")
    if host:
        port = int(os.environ.get("ORCH_SMTP_PORT", "587"))
        user = os.environ.get("ORCH_SMTP_USER")
        password = os.environ.get("ORCH_SMTP_PASS")
        sender = os.environ.get("ORCH_SMTP_FROM") or user or cfg.alert_email

        def _send_smtp(subject: str, body: str) -> None:
            import smtplib
            from email.message import EmailMessage

            msg = EmailMessage()
            msg["From"] = sender
            msg["To"] = cfg.alert_email
            msg["Subject"] = subject
            msg.set_content(body)
            with smtplib.SMTP(host, port, timeout=60) as smtp:
                smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(msg)

        return _send_smtp

    mail = shutil.which("mail") or shutil.which("mailx")
    if not mail:
        logger.info("no `mail` command on PATH and no ORCH_SMTP_HOST; "
                    "skipping email alerts (reports still written to disk)")
        return None

    def _send(subject: str, body: str) -> None:
        subprocess.run(
            [mail, "-s", subject, cfg.alert_email],
            input=body,
            text=True,
            check=True,
            timeout=60,
        )

    return _send
