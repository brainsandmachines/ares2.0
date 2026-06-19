"""In-job progress reporter (runs on the cluster, inside training).

The training loop calls :func:`update_epoch` once per completed epoch. It is a
no-op unless the job was launched by the orchestrator (i.e. ``ORCH_MODEL_ID`` and
``ORCH_DB`` are exported via ``sbatch --export``), so ordinary manual runs are
unaffected. DB hiccups never propagate into training.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .db import OrchestratorDB

logger = logging.getLogger("orchestrator.progress")

_db_cache: dict[str, OrchestratorDB] = {}


def _tracked() -> Optional[tuple[str, str]]:
    model_id = os.environ.get("ORCH_MODEL_ID")
    db_path = os.environ.get("ORCH_DB")
    if model_id and db_path:
        return model_id, db_path
    return None


def _get_db(db_path: str) -> OrchestratorDB:
    db = _db_cache.get(db_path)
    if db is None:
        db = OrchestratorDB(db_path)
        _db_cache[db_path] = db
    return db


def update_epoch(epoch: int, rank: int = 0) -> None:
    """Atomically record ``current_epoch`` for the orchestrated model row."""
    if rank != 0:
        return
    tracked = _tracked()
    if tracked is None:
        return
    model_id, db_path = tracked
    try:
        _get_db(db_path).update_epoch(model_id, int(epoch))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("progress update failed (ignored): %s", exc)
