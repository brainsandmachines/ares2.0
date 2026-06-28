"""In-training DB hooks for the AIRCC job manager.

Imported by the training code (``robust_training/adversarial_training.py`` and
``ares/utils/train_loop.py``). Every function is a **no-op** unless the process
was launched by the job manager, i.e. both ``AIRCC_DB`` and ``AIRCC_MODEL_ID`` are
exported into the environment. This keeps ordinary manual training runs and the
existing orchestrator path completely unaffected. DB hiccups are swallowed so they
never propagate into training.

Links (per the approved plan §11):
  * update_epoch -> current_epoch + heartbeat at the end of every epoch
  * heartbeat    -> heartbeat at every log interval
The post-AA best-checkpoint link is written by ``evaluate_and_plot.py``, not here.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("aircc.progress")

_db_cache: dict[str, object] = {}


def _tracked() -> Optional[tuple[str, str]]:
    model_id = os.environ.get("AIRCC_MODEL_ID")
    db_path = os.environ.get("AIRCC_DB")
    if model_id and db_path:
        return model_id, db_path
    return None


def _get_db(db_path: str):
    db = _db_cache.get(db_path)
    if db is None:
        # Local import so the training process never hard-depends on this package.
        from aircc.aircc_job_manager.db import AirccDB

        db = AirccDB(db_path)
        _db_cache[db_path] = db
    return db


def update_epoch(epoch: int, rank: int = 0, checkpoint: Optional[str] = None) -> None:
    """Record completed-epoch count + heartbeat (rank-0 only; no-op if untracked)."""
    if rank != 0:
        return
    tracked = _tracked()
    if tracked is None:
        return
    model_id, db_path = tracked
    try:
        _get_db(db_path).update_epoch(model_id, int(epoch), checkpoint=checkpoint)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("aircc epoch update failed (ignored): %s", exc)


def heartbeat(rank: int = 0) -> None:
    """Bump the heartbeat (rank-0 only; no-op if untracked)."""
    if rank != 0:
        return
    tracked = _tracked()
    if tracked is None:
        return
    model_id, db_path = tracked
    try:
        _get_db(db_path).heartbeat(model_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("aircc heartbeat failed (ignored): %s", exc)


def write_best_checkpoint(output_dir, rank: int = 0) -> None:
    """Score best/last/advbest by the model's threat model and write it to the DB.

    Called from the in-process final_eval path right after the AA comparison plot.
    The threat model comes from ``AIRCC_THREAT_NORM`` / ``AIRCC_THREAT_EPS`` (empty
    => clean scoring). No-op for untracked runs and non-zero ranks.
    """
    if rank != 0:
        return
    tracked = _tracked()
    if tracked is None:
        return
    model_id, db_path = tracked
    try:
        from pathlib import Path

        from aircc.aircc_job_manager.best_checkpoint import best_checkpoint_for_threat

        norm = os.environ.get("AIRCC_THREAT_NORM", "").strip() or None
        eps_raw = os.environ.get("AIRCC_THREAT_EPS", "").strip()
        eps = float(eps_raw) if eps_raw else None
        path, score = best_checkpoint_for_threat(Path(output_dir), norm, eps)
        if path:
            _get_db(db_path).set_best_checkpoint(model_id, path, score)
            logger.info("aircc best checkpoint %s -> %s (score=%s)", model_id, path, score)
        else:
            logger.warning("aircc best checkpoint: no scored CSV for %s", model_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("aircc best-checkpoint write failed (ignored): %s", exc)
