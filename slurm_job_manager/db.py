"""Operational-state SQLite DB for the Slurm job manager.

Holds **no training hyperparameters** (those live in the CSVs). The DB exists so
concurrent array tasks can atomically claim work, so dead-owner rows can be
requeued, and so a dashboard can report progress.

Schema is a **superset of the aircc ``jobs`` table** so the already-live
in-training hooks in ``aircc.aircc_job_manager.progress`` (``update_epoch``,
``heartbeat``, ``set_best_checkpoint``) write to it unchanged -- the lifecycle
just points ``AIRCC_DB`` at this file. The extra columns
(``slurm_job_id``/``is_test``/``requeued``/``last_error_hash``) plus the
``failure_hashes`` table support liveness-based requeue, the test lane, and
codex-dedup. A lightweight ``ALTER TABLE`` migration adds them regardless of
which side created the base table first.

Ownership invariant: ``slurm_job_id IS NULL`` <=> the row is claimable; a
non-NULL ``slurm_job_id`` means a (possibly dead) array task owns it. Unlike the
aircc manager we do **not** requeue on heartbeat staleness -- requeue is driven
by Slurm liveness (``requeue_dead``) and the SIGTERM-trap ``release``.

All writes run inside one ``BEGIN IMMEDIATE`` transaction with a bounded retry
loop, which is safe over NFS and guarantees two tasks never claim the same row.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Optional

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_FINISHED = "finished"

_BUSY_TIMEOUT_MS = 30_000
_MAX_RETRIES = 8
_RETRY_SLEEP_S = 0.25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    model_name      TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'pending',
    current_epoch   INTEGER NOT NULL DEFAULT 0,
    total_epochs    INTEGER NOT NULL DEFAULT 0,
    heartbeat_ts    INTEGER,
    checkpoint      TEXT,
    best_checkpoint TEXT,
    best_score      REAL,
    priority        INTEGER NOT NULL DEFAULT 100,
    is_test         INTEGER NOT NULL DEFAULT 0,
    slurm_job_id    INTEGER,
    claimed_ts      INTEGER,
    requeued        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    last_error_hash TEXT
);
CREATE TABLE IF NOT EXISTS failure_hashes (
    error_hash     TEXT PRIMARY KEY,
    first_model_id TEXT,
    report_path    TEXT,
    created_ts     INTEGER
);
"""

# Columns added to a pre-existing (e.g. aircc-shaped) ``jobs`` table.
_MIGRATIONS = [
    ("is_test", "is_test INTEGER NOT NULL DEFAULT 0"),
    ("slurm_job_id", "slurm_job_id INTEGER"),
    ("claimed_ts", "claimed_ts INTEGER"),
    ("requeued", "requeued INTEGER NOT NULL DEFAULT 0"),
    ("last_error", "last_error TEXT"),
    ("last_error_hash", "last_error_hash TEXT"),
]


@dataclass
class Job:
    model_name: str
    status: str
    current_epoch: int
    total_epochs: int
    heartbeat_ts: Optional[int]
    checkpoint: Optional[str]
    best_checkpoint: Optional[str]
    best_score: Optional[float]
    priority: int
    is_test: int
    slurm_job_id: Optional[int]
    claimed_ts: Optional[int]
    requeued: int
    last_error: Optional[str]
    last_error_hash: Optional[str]


_JOB_FIELDS = set(Job.__annotations__)


def _row_to_job(row: sqlite3.Row) -> Job:
    # Tolerate extra legacy columns (e.g. aircc's owner_task) by projecting.
    return Job(**{k: row[k] for k in row.keys() if k in _JOB_FIELDS})


def _is_transient(err: sqlite3.OperationalError) -> bool:
    msg = str(err).lower()
    return "locked" in msg or "busy" in msg or "unable to open database file" in msg


class JobDB:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                with self._connect() as conn:
                    conn.executescript(_SCHEMA)
                    existing = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
                    for col, ddl in _MIGRATIONS:
                        if col not in existing:
                            conn.execute(f"ALTER TABLE jobs ADD COLUMN {ddl}")
                    conn.commit()
                return
            except sqlite3.OperationalError as e:
                last_err = e
                if _is_transient(e):
                    time.sleep(_RETRY_SLEEP_S * (attempt + 1))
                    continue
                raise
        raise sqlite3.OperationalError(f"DB init failed after {_MAX_RETRIES} retries: {last_err}")

    # ---- low-level plumbing -------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000.0)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        try:
            yield conn
        finally:
            conn.close()

    def _atomic(self, op):
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = op(conn)
                        conn.commit()
                        return result
                    except Exception:
                        conn.rollback()
                        raise
            except sqlite3.OperationalError as e:
                last_err = e
                if _is_transient(e):
                    time.sleep(_RETRY_SLEEP_S * (attempt + 1))
                    continue
                raise
        raise sqlite3.OperationalError(f"DB write failed after {_MAX_RETRIES} retries: {last_err}")

    def _write(self, sql: str, params: tuple) -> int:
        return self._atomic(lambda conn: conn.execute(sql, params).rowcount)

    # ---- seeding ------------------------------------------------------------
    def upsert_pending(self, model_name: str, total_epochs: int, priority: int,
                       is_test: bool = False) -> None:
        """Insert a fresh pending row; preserve any existing operational state.

        Idempotent: re-seeding never downgrades a running/finished/failed row, it
        only refreshes total_epochs/priority/is_test for an already-present model.
        """
        self._write(
            """
            INSERT INTO jobs (model_name, status, current_epoch, total_epochs, priority, is_test)
            VALUES (?, 'pending', 0, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                total_epochs = excluded.total_epochs,
                priority     = excluded.priority,
                is_test      = excluded.is_test
            """,
            (model_name, int(total_epochs), int(priority), 1 if is_test else 0),
        )

    def reconcile(self, model_name: str, total_epochs: int) -> int:
        """Update total_epochs; re-open a finished row whose progress is short of it.

        Returns 1 if the row was re-opened (finished -> pending), else 0. Lets you
        raise a model's epochs in the CSV and have training resume the delta from
        its own last.pth.tar.
        """
        def op(conn: sqlite3.Connection) -> int:
            conn.execute("UPDATE jobs SET total_epochs=? WHERE model_name=?",
                         (int(total_epochs), model_name))
            cur = conn.execute(
                "UPDATE jobs SET status='pending', slurm_job_id=NULL, claimed_ts=NULL "
                "WHERE model_name=? AND status='finished' AND current_epoch < ?",
                (model_name, int(total_epochs)),
            )
            return cur.rowcount
        return self._atomic(op)

    # ---- claiming -----------------------------------------------------------
    def claim_next(self, slurm_job_id: int, deps: Mapping[str, str]) -> Optional[Job]:
        """Atomically claim the top eligible pending model for this array task.

        Order: **test lane first** (``is_test DESC``), then ``priority ASC`` (lower
        number = sooner). ``deps`` maps model_name -> dependency_model_name (from
        the CSV; '' means none). A dependency is satisfied only when its row is
        FINISHED *and* has a non-empty ``best_checkpoint``. The whole
        SELECT+eligibility+UPDATE runs under one write lock.
        """
        now = int(time.time())

        def op(conn: sqlite3.Connection) -> Optional[Job]:
            candidates = conn.execute(
                "SELECT * FROM jobs WHERE status='pending' AND slurm_job_id IS NULL "
                "ORDER BY is_test DESC, priority ASC"
            ).fetchall()
            for cand in candidates:
                dep = (deps.get(cand["model_name"]) or "").strip()
                if dep:
                    dep_row = conn.execute(
                        "SELECT status, best_checkpoint FROM jobs WHERE model_name=?",
                        (dep,),
                    ).fetchone()
                    if dep_row is None:
                        continue
                    if dep_row["status"] != STATUS_FINISHED:
                        continue
                    if not (dep_row["best_checkpoint"] or "").strip():
                        continue
                conn.execute(
                    "UPDATE jobs SET status='running', slurm_job_id=?, claimed_ts=?, "
                    "heartbeat_ts=? WHERE model_name=? AND slurm_job_id IS NULL",
                    (int(slurm_job_id), now, now, cand["model_name"]),
                )
                fresh = conn.execute(
                    "SELECT * FROM jobs WHERE model_name=?", (cand["model_name"],)
                ).fetchone()
                return _row_to_job(fresh)
            return None

        return self._atomic(op)

    # ---- ownership / requeue ------------------------------------------------
    def running_owned(self) -> list[Job]:
        """Running rows with a Slurm owner (candidates for the liveness check)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status='running' AND slurm_job_id IS NOT NULL"
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def requeue_dead(self, slurm_active: Callable[[int], bool]) -> int:
        """Release running rows whose owning Slurm job is no longer alive.

        ``slurm_active(job_id)`` returns True while the task is present in
        squeue/sacct-active. No heartbeat threshold is involved -- a legitimately
        slow, still-running task is never yanked because its Slurm job is alive.
        """
        n = 0
        for job in self.running_owned():
            if job.slurm_job_id is None:
                continue
            if not slurm_active(int(job.slurm_job_id)):
                self.release(job.model_name)
                n += 1
        return n

    def release(self, model_name: str) -> int:
        """Return a row to the claimable pool (status->pending, clear owner).

        The SIGTERM-trap path (Slurm time-limit) and the dead-owner requeue path.
        Only touches an owned/running row so a concurrent finish/fail wins.
        """
        return self._write(
            "UPDATE jobs SET status='pending', slurm_job_id=NULL, claimed_ts=NULL, "
            "requeued=requeued+1 WHERE model_name=? AND status IN ('running','pending')",
            (model_name,),
        )

    # ---- progress / lifecycle updates --------------------------------------
    def heartbeat(self, model_name: str) -> int:
        return self._write(
            "UPDATE jobs SET heartbeat_ts=? WHERE model_name=?",
            (int(time.time()), model_name),
        )

    def update_epoch(self, model_name: str, epoch: int, checkpoint: Optional[str] = None) -> int:
        return self._write(
            "UPDATE jobs SET current_epoch=?, heartbeat_ts=?, "
            "checkpoint=COALESCE(?, checkpoint) WHERE model_name=?",
            (int(epoch), int(time.time()), checkpoint, model_name),
        )

    def set_best_checkpoint(self, model_name: str, path: str, score: Optional[float]) -> int:
        return self._write(
            "UPDATE jobs SET best_checkpoint=?, best_score=COALESCE(?, best_score), "
            "heartbeat_ts=? WHERE model_name=?",
            (path, None if score is None else float(score), int(time.time()), model_name),
        )

    def mark_finished(self, model_name: str) -> int:
        return self._write(
            "UPDATE jobs SET status='finished', slurm_job_id=NULL, heartbeat_ts=? "
            "WHERE model_name=?",
            (int(time.time()), model_name),
        )

    def mark_failed(self, model_name: str, error: Optional[str] = None,
                    error_hash: Optional[str] = None) -> int:
        return self._write(
            "UPDATE jobs SET status='failed', slurm_job_id=NULL, "
            "last_error=COALESCE(?, last_error), "
            "last_error_hash=COALESCE(?, last_error_hash), heartbeat_ts=? "
            "WHERE model_name=?",
            (None if error is None else str(error)[:8000], error_hash,
             int(time.time()), model_name),
        )

    # ---- reads (no writes) --------------------------------------------------
    def get(self, model_name: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE model_name=?", (model_name,)).fetchone()
        return _row_to_job(row) if row else None

    def get_by_slurm_job_id(self, slurm_job_id: int) -> Optional[Job]:
        """The row currently owned by a Slurm task id (or None). Used by the
        monitor to map a FAILED array task back to its DB row."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE slurm_job_id=?", (int(slurm_job_id),)
            ).fetchone()
        return _row_to_job(row) if row else None

    def all_jobs(self) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY is_test DESC, priority ASC"
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    # ---- failure-hash dedup (codex escalation) ------------------------------
    def has_failure_hash(self, error_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM failure_hashes WHERE error_hash=?", (error_hash,)
            ).fetchone()
        return row is not None

    def record_failure_hash(self, error_hash: str, model_id: str,
                            report_path: Optional[str]) -> int:
        return self._write(
            "INSERT OR IGNORE INTO failure_hashes "
            "(error_hash, first_model_id, report_path, created_ts) VALUES (?, ?, ?, ?)",
            (error_hash, model_id, report_path, int(time.time())),
        )
