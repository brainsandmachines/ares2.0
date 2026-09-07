"""Resumable sha256 cache, keyed on identity rather than path.

Every dedup decision in this package rests on content, not mtime -- because mtime
lies here. 70 AIRCC ``model_best.pth.tar`` files all carry mtime
``2026-08-10 11:12:0x``, written within seconds of each other by a bulk rewrite,
and three of them were confirmed byte-identical to the Slurm copy that a
newest-wins rule would have discarded. So size+mtime is only a cheap pre-filter
and sha256 is the arbiter.

That means hashing on the order of 1 TB, which has to survive a killed tmux
session, so results are cached. The cache key is ``(dev, inode, size, mtime_ns)``,
not the path:

* a **hardlink** into ``models/`` is the same inode, so the curated tree costs zero
  re-hashing;
* a file that is touched or rewritten changes ``mtime_ns`` and is re-hashed, so a
  stale entry can never be served;
* a **rename** (which is how ``pending_deletion`` staging works) keeps the inode,
  so staging does not invalidate the cache either.

Storage is one JSONL append per hashed file on local disk, never on the CIFS share
(``actimeo=1`` and ``flock`` over CIFS are both unreliable).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "slurm_job_manager" / "logs" / "reorg" / ".sha256_cache.jsonl"

# 8 MiB: large enough that a 1.4 GB checkpoint is ~170 reads, small enough not to
# balloon RSS when several passes run in parallel tmux sessions.
CHUNK = 8 * 1024 * 1024


def file_key(path: Path, st: Optional[os.stat_result] = None) -> str:
    """Identity of a file's *content location*: device, inode, size, mtime_ns."""
    st = st or path.stat()
    return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


class HashCache:
    """Append-only JSONL sha256 cache. Safe for one process; lock-guarded per pass."""

    def __init__(self, cache_path: Path = DEFAULT_CACHE):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._entries[rec["key"]] = rec["sha256"]
                except (ValueError, KeyError):
                    # A torn last line from a killed pass -- skip it, the file gets
                    # re-hashed. Never abort the whole run over one bad record.
                    continue

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, path: Path, st: Optional[os.stat_result] = None) -> Optional[str]:
        return self._entries.get(file_key(path, st))

    def sha256(self, path: Path, st: Optional[os.stat_result] = None) -> str:
        """Cached sha256. Computes and appends on a miss."""
        st = st or path.stat()
        key = file_key(path, st)
        cached = self._entries.get(key)
        if cached is not None:
            return cached

        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                block = fh.read(CHUNK)
                if not block:
                    break
                digest.update(block)
        value = digest.hexdigest()

        with self._lock:
            self._entries[key] = value
            with self.cache_path.open("a") as fh:
                fh.write(json.dumps({"key": key, "sha256": value, "path": str(path)}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return value


def same_content(a: Path, b: Path, cache: HashCache) -> bool:
    """True when two files are byte-identical.

    Short-circuits on size (cheap, and a different size is proof of difference)
    and on inode identity (a hardlink pair needs no hashing at all).
    """
    sa, sb = a.stat(), b.stat()
    if sa.st_size != sb.st_size:
        return False
    if (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino):
        return True
    return cache.sha256(a, sa) == cache.sha256(b, sb)
