"""Botero-driven SQLite orchestrator for cluster model training.

Single source of truth = the ``models_queue`` table (see ``db.py``), which lives
on the shared Slurm mount. Cluster jobs are stage-machine executors that read
their row and write atomic per-epoch progress; the botero daemon (``daemon.py``)
polls ``squeue`` over SSH, tops up free GPUs, and runs the MD5-fenced failure
analyzer.
"""
