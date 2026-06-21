"""SQLite access layer for the model orchestrator.

The database (``models_queue``) is the single source of truth and lives on the
shared Slurm mount, so it is written from two places:

* the **botero cron monitor** (tops up array pools, classifies failures), and
* the **cluster array task** itself (atomically *claims* a job, per-epoch
  ``current_epoch`` updates, and status transitions).

NFS POSIX locking is unreliable, so every write goes through ``_write`` (and the
multi-statement claim through ``_atomic``) which:

* opens a short-lived connection with a generous ``busy_timeout``,
* runs inside a ``BEGIN IMMEDIATE`` transaction (reserves the write lock up
  front), and
* retries a bounded number of times on ``database is locked``.

WAL mode is intentionally *not* used (it misbehaves on NFS); we stay on the
default rollback journal with single-row writes to keep contention minimal.

This module is stdlib-only (no pandas/torch) so it imports cleanly inside the
training environment on the cluster.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

# --------------------------------------------------------------------------
# Pipeline status vocabulary (single source of truth for a row's lifecycle).
#
#   PENDING -> TRAINING -> AA_EVAL -> PLOTTING -> FINISHED       (+ FAILED)
#
# Ownership invariant: ``slurm_job_id IS NULL`` means the row is unowned and
# *claimable*; a non-NULL ``slurm_job_id`` means a live array task owns it.
# --------------------------------------------------------------------------
STATUS_PENDING = "PENDING"
STATUS_TRAINING = "TRAINING"
STATUS_AA_EVAL = "AA_EVAL"
STATUS_PLOTTING = "PLOTTING"
STATUS_FINISHED = "FINISHED"
STATUS_FAILED = "FAILED"
STATUSES = (
    STATUS_PENDING,
    STATUS_TRAINING,
    STATUS_AA_EVAL,
    STATUS_PLOTTING,
    STATUS_FINISHED,
    STATUS_FAILED,
)

# Non-terminal statuses a worker can claim, in claim-priority order
# (plotting first, then aa, then training, then a fresh pending row).
CLAIMABLE_STATUSES = (
    STATUS_PLOTTING,
    STATUS_AA_EVAL,
    STATUS_TRAINING,
    STATUS_PENDING,
)
# Statuses that mean "a worker is (supposed to be) actively processing this row"
# -> these are scanned for staleness/requeue.
ACTIVE_STATUSES = (STATUS_TRAINING, STATUS_AA_EVAL, STATUS_PLOTTING)

# Legacy (pre-rebuild) stage values, which are unambiguous markers of an old row
# (the new schema never writes these beyond the default 'TRAIN'): a non-terminal
# legacy row's current_stage tells us where to resume. Used by the one-time,
# idempotent migration in init_db.
_LEGACY_STAGE_TO_STATUS = {
    "AA_SWEEP": STATUS_AA_EVAL,
    "PLOT_RESULTS": STATUS_PLOTTING,
}

# Node groups (Slurm partitions) and their GPU capacities.
NODE_RTX_PRO = "rtx_pro_6000"
NODE_RTX6000 = "rtx6000"
NODE_CAPACITY = {NODE_RTX_PRO: 6, NODE_RTX6000: 8}

_BUSY_TIMEOUT_MS = 30_000
_MAX_RETRIES = 8
_RETRY_SLEEP_S = 0.25


@dataclass(frozen=True)
class ModelRow:
    model_id: str
    status: str
    current_stage: str
    cluster_node: Optional[str]
    slurm_job_id: Optional[int]
    launcher: Optional[str]
    job_name: Optional[str]
    model_dir: Optional[str]
    current_epoch: int
    next_epoch: int
    final_epoch: Optional[int]
    warmup_epochs: int
    linear_epochs: int
    plateau_epochs: int
    requeued: int
    best_checkpoint: Optional[str]
    best_score: Optional[float]
    last_error_hash: Optional[str]
    last_update_ts: Optional[int]
    arch: str = "convnext_small"
    protocol: Optional[str] = None
    eps: Optional[float] = None
    priority: int = 100

    @property
    def total_epochs(self) -> int:
        """Scheduled epochs from the curriculum schedule (warmup+linear+plateau)."""
        return self.warmup_epochs + self.linear_epochs + self.plateau_epochs

    @property
    def target_epoch(self) -> int:
        """Authoritative run length: explicit ``final_epoch`` else the schedule sum."""
        return int(self.final_epoch) if self.final_epoch else self.total_epochs


class OrchestratorDB:
    """Thin, NFS-safe accessor over the ``models_queue`` table."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    # ------------------------------------------------------------------ #
    # Connection / write plumbing
    # ------------------------------------------------------------------ #
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000.0)
        conn.row_factory = sqlite3.Row
        # busy_timeout makes SQLite itself sleep-and-retry on a locked DB; this
        # complements the explicit retry loop below for NFS flakiness.
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        try:
            yield conn
        finally:
            conn.close()

    def _write(self, sql: str, params: tuple = ()) -> int:
        """Run a single write inside BEGIN IMMEDIATE with bounded retries.

        Returns the number of affected rows.
        """

        def op(conn: sqlite3.Connection) -> int:
            return conn.execute(sql, params).rowcount

        return self._atomic(op)

    def _atomic(self, op):
        """Run ``op(conn)`` inside one ``BEGIN IMMEDIATE`` txn, with retries.

        ``op`` may issue several statements (e.g. a SELECT then an UPDATE) and
        returns the value propagated to the caller. The reserved write lock makes
        the whole block atomic against other writers -- the basis of an
        uncontended, race-free job claim across concurrent array tasks.
        """
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
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    time.sleep(_RETRY_SLEEP_S * (attempt + 1))
                    continue
                raise
        raise sqlite3.OperationalError(
            f"DB write failed after {_MAX_RETRIES} retries: {last_err}"
        )

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def init_db(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models_queue (
                    model_id        TEXT PRIMARY KEY,
                    status          TEXT NOT NULL DEFAULT 'PENDING',
                    current_stage   TEXT NOT NULL DEFAULT 'TRAIN',
                    cluster_node    TEXT,
                    slurm_job_id    INTEGER,
                    launcher        TEXT,
                    job_name        TEXT,
                    model_dir       TEXT,
                    current_epoch   INTEGER NOT NULL DEFAULT 0,
                    next_epoch      INTEGER NOT NULL DEFAULT 0,
                    final_epoch     INTEGER,
                    warmup_epochs   INTEGER NOT NULL DEFAULT 0,
                    linear_epochs   INTEGER NOT NULL DEFAULT 0,
                    plateau_epochs  INTEGER NOT NULL DEFAULT 0,
                    requeued        INTEGER NOT NULL DEFAULT 0,
                    best_checkpoint TEXT,
                    best_score      REAL,
                    last_error_hash TEXT,
                    arch            TEXT NOT NULL DEFAULT 'convnext_small',
                    protocol        TEXT,
                    eps             REAL,
                    priority        INTEGER NOT NULL DEFAULT 100,
                    last_update_ts  INTEGER
                )
                """
            )
            # Lightweight migration: add columns to a pre-existing DB.
            existing = {r[1] for r in conn.execute("PRAGMA table_info(models_queue)")}
            for col, ddl in (
                ("next_epoch", "next_epoch INTEGER NOT NULL DEFAULT 0"),
                ("final_epoch", "final_epoch INTEGER"),
                ("requeued", "requeued INTEGER NOT NULL DEFAULT 0"),
                ("best_checkpoint", "best_checkpoint TEXT"),
                ("best_score", "best_score REAL"),
                ("arch", "arch TEXT NOT NULL DEFAULT 'convnext_small'"),
                ("protocol", "protocol TEXT"),
                ("eps", "eps REAL"),
            ):
                if col not in existing:
                    conn.execute(f"ALTER TABLE models_queue ADD COLUMN {ddl}")
            self._migrate_legacy(conn)
            # Global, deterministic dedup store for the failure analyzer: an
            # error signature is processed (LLM + email) at most once, ever.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failure_hashes (
                    error_hash     TEXT PRIMARY KEY,
                    first_model_id TEXT,
                    report_path    TEXT,
                    created_ts     INTEGER
                )
                """
            )
            conn.commit()

    @staticmethod
    def _migrate_legacy(conn: sqlite3.Connection) -> None:
        """Idempotently upgrade pre-rebuild rows to the new pipeline vocabulary.

        Old model: claimable rows were ``PENDING``/``RUNNING`` and ``current_stage``
        said where to resume (TRAIN/AA_SWEEP/PLOT_RESULTS/COMPLETED). We map by
        stage, clear the (now-dead) Slurm owner on non-terminal rows so they are
        immediately claimable, and carry ``current_epoch`` into ``next_epoch`` so
        a resumed job restarts where it left off. PENDING+TRAIN is left untouched
        (it is identical in both schemas), which keeps this safe to run on every
        open.
        """
        # Carry the resume point for legacy rows (new column defaulted to 0).
        conn.execute(
            "UPDATE models_queue SET next_epoch = current_epoch "
            "WHERE next_epoch = 0 AND current_epoch > 0"
        )
        # Terminal rows.
        conn.execute(
            "UPDATE models_queue SET status = ? "
            "WHERE status = 'COMPLETED' OR current_stage = 'COMPLETED'",
            (STATUS_FINISHED,),
        )
        # Non-terminal legacy eval stages -> resume at that stage, release owner.
        for stage, status in _LEGACY_STAGE_TO_STATUS.items():
            conn.execute(
                "UPDATE models_queue SET status = ?, slurm_job_id = NULL "
                "WHERE status IN ('PENDING', 'RUNNING') AND current_stage = ?",
                (status, stage),
            )
        # A row left RUNNING+TRAIN by an old daemon: its job is dead now -> make it
        # claimable as TRAINING (resumes from next_epoch).
        conn.execute(
            "UPDATE models_queue SET status = ?, slurm_job_id = NULL "
            "WHERE status = 'RUNNING' AND current_stage = 'TRAIN'",
            (STATUS_TRAINING,),
        )

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ModelRow:
        return ModelRow(
            model_id=row["model_id"],
            status=row["status"],
            current_stage=row["current_stage"],
            cluster_node=row["cluster_node"],
            slurm_job_id=row["slurm_job_id"],
            launcher=row["launcher"],
            job_name=row["job_name"],
            model_dir=row["model_dir"],
            current_epoch=row["current_epoch"],
            next_epoch=row["next_epoch"],
            final_epoch=row["final_epoch"],
            warmup_epochs=row["warmup_epochs"],
            linear_epochs=row["linear_epochs"],
            plateau_epochs=row["plateau_epochs"],
            requeued=row["requeued"],
            best_checkpoint=row["best_checkpoint"],
            best_score=row["best_score"],
            last_error_hash=row["last_error_hash"],
            last_update_ts=row["last_update_ts"],
            arch=row["arch"],
            protocol=row["protocol"],
            eps=row["eps"],
            priority=row["priority"],
        )

    def get_model_state(self, model_id: str) -> Optional[ModelRow]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models_queue WHERE model_id = ?", (model_id,)
            ).fetchone()
        return self._row_to_model(row) if row else None

    def get_by_slurm_job_id(self, slurm_job_id: int) -> Optional[ModelRow]:
        """Find the row currently owned by a given Slurm task id (or None).

        Used by the monitor to map a FAILED array task back to its DB row; only
        a still-owned row matches, which makes acting on it idempotent (acting
        clears ``slurm_job_id``).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models_queue WHERE slurm_job_id = ?", (int(slurm_job_id),)
            ).fetchone()
        return self._row_to_model(row) if row else None

    def list_models(self, status: Optional[str] = None) -> list[ModelRow]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM models_queue WHERE status = ? "
                    "ORDER BY priority DESC, model_id ASC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM models_queue ORDER BY priority DESC, model_id ASC"
                ).fetchall()
        return [self._row_to_model(r) for r in rows]

    def claimable_count(self, partition: str) -> int:
        """How many rows are currently claimable for ``partition`` (unowned)."""
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) FROM models_queue
                WHERE slurm_job_id IS NULL
                  AND status IN ({','.join('?' * len(CLAIMABLE_STATUSES))})
                  AND (cluster_node IS NULL OR cluster_node = '' OR cluster_node = ?)
                """,
                (*CLAIMABLE_STATUSES, partition),
            ).fetchone()
        return int(row[0])

    def find_stale_candidates(self, thresholds: dict[str, int]) -> list[ModelRow]:
        """Owned (ACTIVE) rows whose idle time exceeds the per-status threshold.

        ``thresholds`` maps status -> seconds. The caller still confirms the
        owning Slurm task is dead (via sacct/squeue) before requeuing.
        """
        now = int(time.time())
        out: list[ModelRow] = []
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM models_queue WHERE slurm_job_id IS NOT NULL "
                f"AND status IN ({','.join('?' * len(ACTIVE_STATUSES))})",
                ACTIVE_STATUSES,
            ).fetchall()
        for r in rows:
            thr = thresholds.get(r["status"])
            if thr is None:
                continue
            last = r["last_update_ts"] or 0
            if now - last > thr:
                out.append(self._row_to_model(r))
        return out

    # ------------------------------------------------------------------ #
    # Atomic claim (the core of the pull model)
    # ------------------------------------------------------------------ #
    def claim_next(self, partition: str, slurm_job_id: int) -> Optional[ModelRow]:
        """Atomically claim the next eligible row for ``partition``.

        Picks the highest-priority *unowned* non-terminal row (PLOTTING >
        AA_EVAL > TRAINING > PENDING, then protocol ``priority`` DESC, then
        ``model_id``), stamps it with ``slurm_job_id``, and -- only for a fresh
        PENDING row -- advances status to TRAINING. Because the whole SELECT +
        UPDATE runs under one ``BEGIN IMMEDIATE`` write lock, two concurrent
        array tasks can never claim the same row.

        Returns the claimed row (with its post-claim status) or None if the
        queue has no eligible work for this partition.
        """
        now = int(time.time())

        def op(conn: sqlite3.Connection) -> Optional[ModelRow]:
            pick = conn.execute(
                f"""
                SELECT * FROM models_queue
                WHERE slurm_job_id IS NULL
                  AND status IN ({','.join('?' * len(CLAIMABLE_STATUSES))})
                  AND (cluster_node IS NULL OR cluster_node = '' OR cluster_node = ?)
                ORDER BY
                  CASE status
                    WHEN 'PLOTTING' THEN 0
                    WHEN 'AA_EVAL'  THEN 1
                    WHEN 'TRAINING' THEN 2
                    WHEN 'PENDING'  THEN 3
                    ELSE 4
                  END,
                  priority DESC,
                  model_id ASC
                LIMIT 1
                """,
                (*CLAIMABLE_STATUSES, partition),
            ).fetchone()
            if pick is None:
                return None
            new_status = (
                STATUS_TRAINING if pick["status"] == STATUS_PENDING else pick["status"]
            )
            conn.execute(
                "UPDATE models_queue SET slurm_job_id = ?, cluster_node = ?, "
                "status = ?, last_update_ts = ? "
                "WHERE model_id = ? AND slurm_job_id IS NULL",
                (int(slurm_job_id), partition, new_status, now, pick["model_id"]),
            )
            # Re-read so the returned row reflects the post-claim state.
            fresh = conn.execute(
                "SELECT * FROM models_queue WHERE model_id = ?", (pick["model_id"],)
            ).fetchone()
            return self._row_to_model(fresh)

        return self._atomic(op)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert_model(
        self,
        model_id: str,
        launcher: str,
        job_name: str,
        model_dir: str,
        warmup_epochs: int,
        linear_epochs: int,
        plateau_epochs: int,
        cluster_node: Optional[str] = None,
        priority: int = 100,
        arch: str = "convnext_small",
        protocol: Optional[str] = None,
        eps: Optional[float] = None,
        final_epoch: Optional[int] = None,
    ) -> None:
        """Insert a queue row, leaving an existing row's runtime state intact.

        ``final_epoch`` defaults to the curriculum schedule sum but can be set
        explicitly to override the desired endpoint.
        """
        if final_epoch is None:
            final_epoch = int(warmup_epochs) + int(linear_epochs) + int(plateau_epochs)
        self._write(
            """
            INSERT INTO models_queue
                (model_id, status, current_stage, cluster_node, launcher,
                 job_name, model_dir, warmup_epochs, linear_epochs,
                 plateau_epochs, final_epoch, arch, protocol, eps, priority,
                 last_update_ts)
            VALUES (?, 'PENDING', 'TRAIN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                launcher       = excluded.launcher,
                job_name       = excluded.job_name,
                model_dir      = excluded.model_dir,
                warmup_epochs  = excluded.warmup_epochs,
                linear_epochs  = excluded.linear_epochs,
                plateau_epochs = excluded.plateau_epochs,
                final_epoch    = excluded.final_epoch,
                arch           = excluded.arch,
                protocol       = excluded.protocol,
                eps            = excluded.eps,
                cluster_node   = excluded.cluster_node,
                priority       = excluded.priority
            """,
            (
                model_id,
                cluster_node,
                launcher,
                job_name,
                model_dir,
                int(warmup_epochs),
                int(linear_epochs),
                int(plateau_epochs),
                int(final_epoch),
                arch,
                protocol,
                None if eps is None else float(eps),
                int(priority),
                int(time.time()),
            ),
        )

    def update_epoch(self, model_id: str, epoch: int) -> int:
        """Atomic per-epoch progress update (called from the training job).

        The training loop passes ``epoch+1`` (the count of completed epochs),
        which is also exactly the epoch to *resume* at, so we mirror it into
        ``next_epoch``.
        """
        return self._write(
            "UPDATE models_queue SET current_epoch = ?, next_epoch = ?, "
            "last_update_ts = ? WHERE model_id = ?",
            (int(epoch), int(epoch), int(time.time()), model_id),
        )

    def set_status(self, model_id: str, status: str) -> int:
        if status not in STATUSES:
            raise ValueError(f"invalid status: {status}")
        return self._write(
            "UPDATE models_queue SET status = ?, last_update_ts = ? WHERE model_id = ?",
            (status, int(time.time()), model_id),
        )

    def requeue(self, model_id: str) -> int:
        """Release ownership so the row becomes claimable again (status kept).

        Used both by the in-job stale scan and the monitor's preemption path.
        Bumps the ``requeued`` counter so requeues are visible/auditable.
        """
        return self._write(
            "UPDATE models_queue SET slurm_job_id = NULL, requeued = requeued + 1, "
            "last_update_ts = ? WHERE model_id = ?",
            (int(time.time()), model_id),
        )

    def mark_failed(self, model_id: str, error_hash: Optional[str] = None) -> int:
        return self._write(
            "UPDATE models_queue SET status = 'FAILED', slurm_job_id = NULL, "
            "last_error_hash = COALESCE(?, last_error_hash), last_update_ts = ? "
            "WHERE model_id = ?",
            (error_hash, int(time.time()), model_id),
        )

    def set_best_checkpoint(
        self, model_id: str, kind: str, score: Optional[float] = None
    ) -> int:
        return self._write(
            "UPDATE models_queue SET best_checkpoint = ?, "
            "best_score = COALESCE(?, best_score), last_update_ts = ? "
            "WHERE model_id = ?",
            (kind, None if score is None else float(score), int(time.time()), model_id),
        )

    def mark_finished(
        self,
        model_id: str,
        best_checkpoint: Optional[str] = None,
        best_score: Optional[float] = None,
    ) -> int:
        """Terminal success: status FINISHED, release ownership, record the
        winning checkpoint kind and its score."""
        return self._write(
            "UPDATE models_queue SET status = 'FINISHED', slurm_job_id = NULL, "
            "best_checkpoint = COALESCE(?, best_checkpoint), "
            "best_score = COALESCE(?, best_score), last_update_ts = ? "
            "WHERE model_id = ?",
            (
                best_checkpoint,
                None if best_score is None else float(best_score),
                int(time.time()),
                model_id,
            ),
        )

    def force_status(self, model_id: str, status: str, release: bool = True) -> int:
        """Operator override: jump a row to ``status`` (and clear ownership)."""
        if status not in STATUSES:
            raise ValueError(f"invalid status: {status}")
        owner = "slurm_job_id = NULL, " if release else ""
        return self._write(
            f"UPDATE models_queue SET status = ?, {owner}last_update_ts = ? "
            "WHERE model_id = ?",
            (status, int(time.time()), model_id),
        )

    # ------------------------------------------------------------------ #
    # Failure-hash dedup store
    # ------------------------------------------------------------------ #
    def has_failure_hash(self, error_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM failure_hashes WHERE error_hash = ?", (error_hash,)
            ).fetchone()
        return row is not None

    def record_failure_hash(
        self, error_hash: str, model_id: str, report_path: Optional[str]
    ) -> int:
        return self._write(
            "INSERT OR IGNORE INTO failure_hashes "
            "(error_hash, first_model_id, report_path, created_ts) "
            "VALUES (?, ?, ?, ?)",
            (error_hash, model_id, report_path, int(time.time())),
        )
