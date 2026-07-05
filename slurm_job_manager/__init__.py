"""Simple, adaptive Slurm job manager for the Botero cluster.

CSV is the single source of truth (one column per Hydra override + metadata);
the SQLite DB holds operational state only. One array task = one GPU = one model:
requeue dead owners -> atomically claim one eligible row -> run a single-process
``adversarial_training`` lifecycle (train -> final_eval AA -> plot) -> mark
finished, or on failure classify -> requeue/mark_failed. A SIGTERM trap in the
sbatch releases a row on Slurm time-limit so the next task resumes it.

Replaces ``orchestrator/`` (left in place, unused). Reuses the already-live
``aircc.aircc_job_manager.progress`` in-training hooks via ``AIRCC_DB`` /
``AIRCC_MODEL_ID``.
"""
