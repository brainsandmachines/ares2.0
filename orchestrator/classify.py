"""Deterministic failure classification for orchestrated jobs.

A job's log tail is mapped to one of three actions by an ordered rule table.
*Everything here is deterministic* -- only logs that match no rule come back as
``ACTION_UNKNOWN``, which the botero monitor escalates to the ``codex`` CLI for a
one-off human-readable diagnosis (deduped by error hash so each new signature is
escalated exactly once).

Shared by:
* ``controller.py`` (cluster): decide requeue-vs-fail for the job it just ran.
* ``monitor.py`` (botero): classify failed array tasks, escalate unknowns.
"""

from __future__ import annotations

import hashlib
import re

# Actions ------------------------------------------------------------------
ACTION_REQUEUE = "requeue"   # transient / preemption / OOM -> release & retry
ACTION_FAIL = "fail"         # deterministic code/config error -> mark FAILED
ACTION_UNKNOWN = "unknown"   # unrecognized -> monitor escalates to codex

# Ordered (regex, action) rules. First match wins; the regexes are matched
# case-insensitively against the job's log tail. Order matters: preemption /
# timeout signals are checked before generic tracebacks so a job killed by the
# scheduler is retried rather than marked failed.
_RULES: list[tuple[re.Pattern, str]] = [
    # --- transient / scheduler -> requeue (retry) ---
    (re.compile(r"DUE TO TIME LIMIT", re.I), ACTION_REQUEUE),
    (re.compile(r"CANCELLED AT .* DUE TO (PREEMPTION|NODE FAILURE)", re.I), ACTION_REQUEUE),
    (re.compile(r"slurmstepd: error: .*(oom-kill|Out Of Memory)", re.I), ACTION_REQUEUE),
    (re.compile(r"CUDA out of memory", re.I), ACTION_REQUEUE),
    (re.compile(r"CUDA error: out of memory", re.I), ACTION_REQUEUE),
    (re.compile(r"NCCL .*(timeout|unhandled|remote process exited)", re.I), ACTION_REQUEUE),
    (re.compile(r"Bus error|Segmentation fault", re.I), ACTION_REQUEUE),
    (re.compile(r"DataLoader worker .* killed", re.I), ACTION_REQUEUE),
    (re.compile(r"(Connection reset by peer|Broken pipe|Stale file handle)", re.I), ACTION_REQUEUE),
    # --- deterministic code / config errors -> fail ---
    (re.compile(r"ModuleNotFoundError|ImportError", re.I), ACTION_FAIL),
    (re.compile(r"FileNotFoundError|No such file or directory", re.I), ACTION_FAIL),
    (re.compile(r"hydra\.errors\.|omegaconf\..*Error|ConfigAttributeError", re.I), ACTION_FAIL),
    (re.compile(r"KeyError|ValueError|TypeError|AttributeError|IndexError", re.I), ACTION_FAIL),
    (re.compile(r"AssertionError", re.I), ACTION_FAIL),
    (re.compile(r"size mismatch|shapes? .* (are|is) (not|in)compatible", re.I), ACTION_FAIL),
]

_TB_MARK = "Traceback (most recent call last):"


def classify(log_text: str) -> str:
    """Map a log tail to an action. Deterministic; unmatched -> ACTION_UNKNOWN.

    An empty/missing log (killed before writing a traceback) is treated as a
    transient interruption -> requeue, never a hard failure.
    """
    if not log_text or not log_text.strip():
        return ACTION_REQUEUE
    for pattern, action in _RULES:
        if pattern.search(log_text):
            return action
    # A clean exit with no traceback is almost always a timeout/preemption.
    if _TB_MARK not in log_text:
        return ACTION_REQUEUE
    return ACTION_UNKNOWN


def extract_traceback(log_text: str) -> str:
    """Return the last Python traceback in the log (or "" if none)."""
    if not log_text or _TB_MARK not in log_text:
        return ""
    return _TB_MARK + log_text.rsplit(_TB_MARK, 1)[1]


def error_hash(log_text: str) -> str:
    """Stable signature for dedup: hash the normalized traceback (paths/addrs/numbers stripped).

    Falls back to a hash of the whole (normalized) log when there is no traceback,
    so timeout/preemption logs still dedup sanely.
    """
    tb = extract_traceback(log_text) or (log_text or "")
    norm = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", tb)
    norm = re.sub(r"line \d+", "line N", norm)
    norm = re.sub(r"/[^\s'\":]+", "<PATH>", norm)   # whole absolute-path tokens
    norm = re.sub(r"\d+", "N", norm)
    return hashlib.md5(norm.encode("utf-8", "replace")).hexdigest()
