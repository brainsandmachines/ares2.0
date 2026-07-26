"""AIRCC monitoring/claiming SQLite database.

Holds **operational state only** -- no training hyperparameters (those live in the
CSVs). The DB exists so parallel job-manager slots can atomically claim work, so
stale jobs can be requeued, and so a dashboard can report progress.

All writes run inside a single ``BEGIN IMMEDIATE`` transaction with a bounded
retry loop, which is safe over NFS and guarantees two slots/array-tasks can never
claim the same model. Dependency information is NOT stored here; the caller passes
a ``{model_name -> dependency_model_name}`` map (read from the CSV) into
:meth:`claim_next`, and the eligibility check runs inside the held write lock.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping, Optional

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_FINISHED = "finished"

STALE_SECONDS = 15 * 60  # 15 minutes without a heartbeat -> requeue

_BUSY_TIMEOUT_MS = 30_000
_MAX_RETRIES = 8
_RETRY_SLEEP_S = 0.25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    model_name      TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'pending',
    current_epoch   INTEGER NOT NULL DEFAULT 0,
    total_epochs    INTEGER NOT NULL,
    heartbeat_ts    INTEGER,
    checkpoint      TEXT,
    best_checkpoint TEXT,
    best_score      REAL,
    priority        INTEGER NOT NULL,
    owner_task      INTEGER,
    claimed_ts      INTEGER,
    last_error      TEXT
);
"""


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
    owner_task: Optional[int]
    claimed_ts: Optional[int]
    last_error: Optional[str]


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(**{k: row[k] for k in row.keys()})


def _is_transient(err: sqlite3.OperationalError) -> bool:
    msg = str(err).lower()
    return "locked" in msg or "busy" in msg or "unable to open database file" in msg


class AirccDB:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                with self._connect() as conn:
                    conn.executescript(_SCHEMA)
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
        """Run ``op(conn)`` inside one BEGIN IMMEDIATE txn, retrying on lock/busy."""
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
    def upsert_pending(self, model_name: str, total_epochs: int, priority: int) -> None:
        """Insert a fresh pending row; preserve any existing operational state.

        Idempotent: re-seeding never downgrades a running/finished/failed row, it
        only fills total_epochs/priority for an already-present model.
        """
        self._write(
            """
            INSERT INTO jobs (model_name, status, current_epoch, total_epochs, priority)
            VALUES (?, 'pending', 0, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                total_epochs = excluded.total_epochs,
                priority     = excluded.priority
            """,
            (model_name, int(total_epochs), int(priority)),
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
                "UPDATE jobs SET status='pending', owner_task=NULL, claimed_ts=NULL "
                "WHERE model_name=? AND status='finished' AND current_epoch < ?",
                (model_name, int(total_epochs)),
            )
            return cur.rowcount
        return self._atomic(op)

    def sync_pending_spec(self, spec: Mapping[str, tuple[int, int]]) -> list[str]:
        """Push CSV ``(total_epochs, priority)`` onto rows still pending + unclaimed.

        ``spec`` maps model_name -> (total_epochs, priority). Called before every
        claim so a CSV priority edit reorders the queue live (claim_next orders by
        the DB column, not the CSV). Returns the names actually changed.

        Deliberately UPDATE-only: seeding is selective (--arch/--init) while the
        CSVs hold every model, so inserting here would pull unseeded models into
        the queue. Running/finished/failed rows and parked rows (owner_task=-1)
        are left alone; re-opening a finished row stays a seed_db --reconcile job.
        """
        def op(conn: sqlite3.Connection) -> list[str]:
            cur = conn.execute(
                "SELECT model_name, total_epochs, priority FROM jobs "
                "WHERE status='pending' AND owner_task IS NULL"
            )
            changed: list[str] = []
            for row in cur.fetchall():
                want = spec.get(row["model_name"])
                if want is None:
                    continue
                total, prio = int(want[0]), int(want[1])
                if row["total_epochs"] == total and row["priority"] == prio:
                    continue
                conn.execute(
                    "UPDATE jobs SET total_epochs=?, priority=? WHERE model_name=? "
                    "AND status='pending' AND owner_task IS NULL",
                    (total, prio, row["model_name"]),
                )
                changed.append(row["model_name"])
            return changed

        return self._atomic(op)

    # ---- claiming -----------------------------------------------------------
    def claim_next(self, owner_task: int, deps: Mapping[str, str]) -> Optional[Job]:
        """Atomically claim the highest-priority eligible pending model.

        ``deps`` maps model_name -> dependency_model_name (from the CSV; '' / missing
        means no dependency). A dependency is satisfied only when the dependency row
        is FINISHED *and* has a non-empty ``best_checkpoint``. The whole
        SELECT+eligibility+UPDATE runs under one write lock, so no model is ever
        claimed twice.
        """
        now = int(time.time())

        def op(conn: sqlite3.Connection) -> Optional[Job]:
            candidates = conn.execute(
                "SELECT * FROM jobs WHERE status='pending' AND owner_task IS NULL "
                "ORDER BY priority ASC"
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
                    "UPDATE jobs SET status='running', owner_task=?, claimed_ts=?, "
                    "heartbeat_ts=? WHERE model_name=? AND owner_task IS NULL",
                    (int(owner_task), now, now, cand["model_name"]),
                )
                fresh = conn.execute(
                    "SELECT * FROM jobs WHERE model_name=?", (cand["model_name"],)
                ).fetchone()
                return _row_to_job(fresh)
            return None

        return self._atomic(op)

    # ---- stale handling -----------------------------------------------------
    def requeue_stale(self, stale_seconds: int = STALE_SECONDS) -> int:
        """Reset running rows with no heartbeat for >stale_seconds back to pending."""
        cutoff = int(time.time()) - int(stale_seconds)
        return self._write(
            "UPDATE jobs SET status='pending', owner_task=NULL, claimed_ts=NULL "
            "WHERE status='running' AND (heartbeat_ts IS NULL OR heartbeat_ts < ?)",
            (cutoff,),
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
            "UPDATE jobs SET status='finished', owner_task=NULL, heartbeat_ts=? "
            "WHERE model_name=?",
            (int(time.time()), model_name),
        )

    def mark_failed(self, model_name: str, error: str) -> int:
        return self._write(
            "UPDATE jobs SET status='failed', owner_task=NULL, last_error=?, heartbeat_ts=? "
            "WHERE model_name=?",
            (str(error)[:2000], int(time.time()), model_name),
        )

    # ---- reads (no writes) --------------------------------------------------
    def get(self, model_name: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE model_name=?", (model_name,)).fetchone()
        return _row_to_job(row) if row else None

    def all_jobs(self) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY priority ASC").fetchall()
        return [_row_to_job(r) for r in rows]
