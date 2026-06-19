"""Deterministic, MD5-fenced failure analysis.

When a job fails, the orchestrator extracts the traceback, normalizes away all
variable tokens (timestamps, paths, addresses, ids), and hashes the structural
remainder. A given signature is escalated to the LLM analyzer + email **at most
once, ever** (dedup via ``failure_hashes``). Repeat failures with the same root
cause are silently skipped — this is the bound that keeps the LLM step from
looping or spamming.

The LLM client and emailer are injected callables so this module is testable
offline and has no hard external dependency.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .db import OrchestratorDB

logger = logging.getLogger("orchestrator.failure")

# llm_client(raw_traceback: str) -> markdown_report: str
LLMClient = Callable[[str], str]
# emailer(subject: str, body: str) -> None
Emailer = Callable[[str, str], None]

# --- normalization patterns (order matters: specific before generic) ---
_SUBS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),                       # memory addrs
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\S*"), "<TS>"),  # timestamps
    (re.compile(r"\bline \d+\b"), "line <N>"),                      # source line nums
    (re.compile(r"/[^\s'\":]+"), "<PATH>"),                         # absolute paths
    (re.compile(r"\b\d+\b"), "<N>"),                                # any bare number
]

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")


@dataclass
class FailureOutcome:
    error_hash: str
    is_new: bool
    report_path: Optional[str] = None


def extract_traceback(log_text: str, tail_lines: int = 80) -> str:
    """Return the last Python traceback block, or the log tail as a fallback."""
    matches = list(_TRACEBACK_RE.finditer(log_text))
    if matches:
        return log_text[matches[-1].start():].strip()
    return "\n".join(log_text.splitlines()[-tail_lines:]).strip()


def normalize_traceback(text: str) -> str:
    """Strip variable tokens so structurally-identical errors hash identically."""
    norm = text
    for pattern, repl in _SUBS:
        norm = pattern.sub(repl, norm)
    # collapse whitespace so formatting noise doesn't affect the hash
    norm = re.sub(r"[ \t]+", " ", norm)
    norm = "\n".join(line.strip() for line in norm.splitlines() if line.strip())
    return norm


def traceback_hash(text: str) -> str:
    return hashlib.md5(normalize_traceback(text).encode("utf-8")).hexdigest()


def analyze_failure(
    db: OrchestratorDB,
    model_id: str,
    log_text: str,
    recommendations_dir: str,
    llm_client: Optional[LLMClient] = None,
    emailer: Optional[Emailer] = None,
) -> FailureOutcome:
    """Process one failed job. Deterministic + idempotent via the hash store."""
    raw_tb = extract_traceback(log_text)
    err_hash = traceback_hash(raw_tb)

    # Always record the hash on the row (cheap, informative).
    db.mark_failed(model_id, err_hash)

    if db.has_failure_hash(err_hash):
        logger.info("failure %s already seen; skipping LLM/email for %s",
                    err_hash, model_id)
        return FailureOutcome(error_hash=err_hash, is_new=False)

    # New signature -> escalate exactly once.
    report_path = os.path.join(recommendations_dir, f"{err_hash}.md")
    os.makedirs(recommendations_dir, exist_ok=True)

    report_md = None
    if llm_client is not None:
        try:
            report_md = llm_client(raw_tb)
        except Exception as exc:  # never let the analyzer crash the loop
            logger.exception("LLM client failed for %s: %s", model_id, exc)

    if report_md is None:
        report_md = (
            f"# Failure report\n\n"
            f"- model_id: `{model_id}`\n- signature: `{err_hash}`\n"
            f"- time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"## Raw traceback\n\n```\n{raw_tb}\n```\n"
        )
    with open(report_path, "w") as fh:
        fh.write(report_md)

    if emailer is not None:
        try:
            emailer(
                f"[orchestrator] new failure signature {err_hash[:8]} ({model_id})",
                report_md,
            )
        except Exception as exc:
            logger.exception("emailer failed for %s: %s", model_id, exc)

    db.record_failure_hash(err_hash, model_id, report_path)
    logger.info("recorded NEW failure signature %s for %s -> %s",
                err_hash, model_id, report_path)
    return FailureOutcome(error_hash=err_hash, is_new=True, report_path=report_path)
