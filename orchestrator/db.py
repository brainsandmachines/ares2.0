"""SQLite access layer for the model orchestrator.

The database (``models_queue``) is the single source of truth and lives on the
shared Slurm mount, so it is written from two places:

* the **botero orchestrator** (submits jobs, tops up GPUs, marks failures), and
* the **cluster job** itself (atomic per-epoch ``current_epoch`` updates and
  stage transitions).

NFS POSIX locking is unreliable, so every write goes through ``_write`` which:

* opens a short-lived connection with a generous ``busy_timeout``,
* runs inside a ``BEGIN IMMEDIATE`` transaction (reserves the write lock up
  front, the same pattern the legacy ``claim_next_waiting_model`` relied on), and
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
# Status / stage vocabulary (kept as plain strings for easy SQL + logging).
# --------------------------------------------------------------------------
STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED)

STAGE_TRAIN = "TRAIN"
STAGE_AA_SWEEP = "AA_SWEEP"
STAGE_PLOT_RESULTS = "PLOT_RESULTS"
STAGE_CUSTOM_TASK = "CUSTOM_TASK"
STAGE_COMPLETED = "COMPLETED"

# Deterministic forward pipeline. CUSTOM_TASK is optional and only entered when
# explicitly set; the default happy path is TRAIN -> AA_SWEEP -> PLOT -> done.
STAGE_PIPELINE = (STAGE_TRAIN, STAGE_AA_SWEEP, STAGE_PLOT_RESULTS, STAGE_COMPLETED)

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
    warmup_epochs: int
    linear_epochs: int
    plateau_epochs: int
    last_error_hash: Optional[str]
    arch: str = "convnext_small"
    protocol: Optional[str] = None
    eps: Optional[float] = None

    @property
    def total_epochs(self) -> int:
        """Total scheduled epochs = warmup + linear + plateau."""
        return self.warmup_epochs + self.linear_epochs + self.plateau_epochs


def next_stage(stage: str) -> str:
    """Return the next stage in the deterministic pipeline (idempotent at end)."""
    if stage not in STAGE_PIPELINE:
        # CUSTOM_TASK or anything off-pipeline collapses to COMPLETED next.
        return STAGE_COMPLETED
    idx = STAGE_PIPELINE.index(stage)
    return STAGE_PIPELINE[min(idx + 1, len(STAGE_PIPELINE) - 1)]


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
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    cur = conn.execute(sql, params)
                    conn.commit()
                    return cur.rowcount
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
                    warmup_epochs   INTEGER NOT NULL DEFAULT 0,
                    linear_epochs   INTEGER NOT NULL DEFAULT 0,
                    plateau_epochs  INTEGER NOT NULL DEFAULT 0,
                    last_error_hash TEXT,
                    arch            TEXT NOT NULL DEFAULT 'convnext_small',
                    protocol        TEXT,
                    eps             REAL,
                    priority        INTEGER NOT NULL DEFAULT 100,
                    last_update_ts  INTEGER
                )
                """
            )
            # Lightweight migration: add descriptive columns to a pre-existing DB.
            existing = {r[1] for r in conn.execute("PRAGMA table_info(models_queue)")}
            for col, ddl in (
                ("arch", "arch TEXT NOT NULL DEFAULT 'convnext_small'"),
                ("protocol", "protocol TEXT"),
                ("eps", "eps REAL"),
            ):
                if col not in existing:
                    conn.execute(f"ALTER TABLE models_queue ADD COLUMN {ddl}")
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
            warmup_epochs=row["warmup_epochs"],
            linear_epochs=row["linear_epochs"],
            plateau_epochs=row["plateau_epochs"],
            last_error_hash=row["last_error_hash"],
            arch=row["arch"],
            protocol=row["protocol"],
            eps=row["eps"],
        )

    def get_model_state(self, model_id: str) -> Optional[ModelRow]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models_queue WHERE model_id = ?", (model_id,)
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

    def next_pending_for_node(
        self,
        node: str,
        exclude: Optional[set] = None,
    ) -> Optional["ModelRow"]:
        """Deterministically pick the next PENDING row eligible for ``node``.

        A row is eligible if its ``cluster_node`` is unset (any node) or matches.
        Ordering: eval stages before training (PLOT_RESULTS→AA_SWEEP→TRAIN),
        then protocol priority DESC (higher number = higher priority), then
        model_id for a stable tiebreak.
        ``exclude`` is an optional set of model_ids already claimed this tick.
        """
        exclude_ids = list(exclude) if exclude else []
        ex_clause = (
            f"AND model_id NOT IN ({','.join('?' * len(exclude_ids))})"
            if exclude_ids
            else ""
        )
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM models_queue
                WHERE status = 'PENDING'
                  AND (cluster_node IS NULL OR cluster_node = '' OR cluster_node = ?)
                  {ex_clause}
                ORDER BY
                  CASE current_stage
                    WHEN 'PLOT_RESULTS' THEN 0
                    WHEN 'AA_SWEEP'     THEN 1
                    WHEN 'TRAIN'        THEN 2
                    ELSE 3
                  END,
                  priority DESC,
                  model_id ASC
                LIMIT 1
                """,
                (node, *exclude_ids),
            ).fetchone()
        return self._row_to_model(row) if row else None

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
    ) -> None:
        """Insert a queue row, leaving an existing row's runtime state intact."""
        self._write(
            """
            INSERT INTO models_queue
                (model_id, status, current_stage, cluster_node, launcher,
                 job_name, model_dir, warmup_epochs, linear_epochs,
                 plateau_epochs, arch, protocol, eps, priority, last_update_ts)
            VALUES (?, 'PENDING', 'TRAIN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                launcher       = excluded.launcher,
                job_name       = excluded.job_name,
                model_dir      = excluded.model_dir,
                warmup_epochs  = excluded.warmup_epochs,
                linear_epochs  = excluded.linear_epochs,
                plateau_epochs = excluded.plateau_epochs,
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
                arch,
                protocol,
                None if eps is None else float(eps),
                int(priority),
                int(time.time()),
            ),
        )

    def update_epoch(self, model_id: str, epoch: int) -> int:
        """Atomic single-row per-epoch progress update (called from the job)."""
        return self._write(
            "UPDATE models_queue SET current_epoch = ?, last_update_ts = ? "
            "WHERE model_id = ?",
            (int(epoch), int(time.time()), model_id),
        )

    def advance_stage(self, model_id: str, stage: str) -> int:
        """Set ``current_stage`` (and mark COMPLETED status when appropriate)."""
        status = STATUS_COMPLETED if stage == STAGE_COMPLETED else STATUS_RUNNING
        return self._write(
            "UPDATE models_queue SET current_stage = ?, status = ?, "
            "last_update_ts = ? WHERE model_id = ?",
            (stage, status, int(time.time()), model_id),
        )

    def mark_running(self, model_id: str, slurm_job_id: int, node: str) -> int:
        """Botero-side: a job was just submitted for this row."""
        return self._write(
            "UPDATE models_queue SET status = 'RUNNING', slurm_job_id = ?, "
            "cluster_node = ?, last_update_ts = ? WHERE model_id = ?",
            (int(slurm_job_id), node, int(time.time()), model_id),
        )

    def mark_failed(self, model_id: str, error_hash: Optional[str] = None) -> int:
        return self._write(
            "UPDATE models_queue SET status = 'FAILED', last_error_hash = "
            "COALESCE(?, last_error_hash), last_update_ts = ? WHERE model_id = ?",
            (error_hash, int(time.time()), model_id),
        )

    def set_status(self, model_id: str, status: str) -> int:
        if status not in STATUSES:
            raise ValueError(f"invalid status: {status}")
        return self._write(
            "UPDATE models_queue SET status = ?, last_update_ts = ? WHERE model_id = ?",
            (status, int(time.time()), model_id),
        )

    def force_stage(self, model_id: str, stage: str, status: str = STATUS_PENDING) -> int:
        """Operator override: jump a row to a stage and (re)queue it."""
        return self._write(
            "UPDATE models_queue SET current_stage = ?, status = ?, "
            "last_update_ts = ? WHERE model_id = ?",
            (stage, status, int(time.time()), model_id),
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

    def requeue_stale_running(self, threshold_seconds: int) -> int:
        """Reset RUNNING rows with no progress past the threshold back to PENDING."""
        cutoff = int(time.time()) - int(threshold_seconds)
        return self._write(
            "UPDATE models_queue SET status = 'PENDING', slurm_job_id = NULL "
            "WHERE status = 'RUNNING' "
            "AND (last_update_ts IS NULL OR last_update_ts < ?)",
            (cutoff,),
        )
