"""Curated model store: one canonical home for every trained checkpoint.

This package owns three trees and nothing else:

* ``/mnt/botero/{aircc,slurm}_archive`` -- the QNAP master archive. **Read-only here.**
  Written only by ``slurm_job_manager/scripts/backup_slurm_models.sh`` and
  ``scripts/mirror_archives_to_qnap.sh``, both append-only.
* ``/mnt/data4t/models`` -- the curated working copy: every model, every keeper
  checkpoint (last / model_best / model_best_adv / periodic), **no** ``checkpoint-N``.
* ``/mnt/data4t/models_for_experiments`` -- ``<arch>/<protocol>/<norm>/<name>.pth.tar``
  symlinks to the blessed checkpoint. The single root ``epsilon_bounded_contstim``
  reads from.

Why a separate package rather than a home in either job manager: it spans both
(``slurm_job_manager`` and ``aircc/aircc_job_manager``) plus ``/mnt/data``, and it
outlives both -- the AIRCC campaign is frozen and its DB is now a snapshot file.

Every pass in here is **dry-run by default** and needs ``--apply``, is idempotent,
and is resumable. Nothing in this package deletes: removals are a same-filesystem
``mv`` into ``pending_deletion/``, which the user erases by hand. The single
exception is the ``--delete`` that prunes stale *symlinks* from
``models_for_experiments`` -- that destroys no checkpoint data.
"""
