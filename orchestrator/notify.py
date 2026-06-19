"""Pluggable LLM-analyzer and email transports for the failure analyzer.

Both are best-effort and optional: if the ``claude`` CLI or a mail transport
isn't available, the analyzer still writes the raw-traceback markdown report and
the loop keeps running. Nothing here is on the critical scheduling path.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Callable, Optional

from .config import Config

logger = logging.getLogger("orchestrator.notify")

_LLM_PROMPT = (
    "You are debugging a failed ML training job on a SLURM cluster (ARES, "
    "adversarial robustness, PyTorch/Hydra). Given the traceback below, produce "
    "a concise markdown report with: (1) one-line root cause, (2) the most "
    "likely fix, (3) any config/launcher lines to change. Be specific and short.\n\n"
    "Traceback:\n```\n{tb}\n```\n"
)


def make_llm_client() -> Optional[Callable[[str], str]]:
    """Return an LLM client backed by the local ``claude`` CLI, or None."""
    claude = shutil.which("claude")
    if not claude:
        logger.info("no `claude` CLI on PATH; failure reports will be raw-only")
        return None

    def _client(traceback_text: str) -> str:
        prompt = _LLM_PROMPT.format(tb=traceback_text[:8000])
        proc = subprocess.run(
            [claude, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {proc.stderr[:500]}")
        return proc.stdout.strip()

    return _client


def make_emailer(cfg: Config) -> Optional[Callable[[str, str], None]]:
    """Return an emailer using the local ``mail`` command, or None."""
    if not cfg.alert_email:
        return None
    mail = shutil.which("mail") or shutil.which("mailx")
    if not mail:
        logger.info("no `mail` command on PATH; skipping email alerts")
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
