"""Prefer the local Botero mirror over the AIRCC sshfs mount as the staging source.

The 03:00 backup cron (`aircc/aircc_job_manager/scripts/backup_aircc_models.sh`) already pulls the
AIRCC results tree to `/mnt/data4t/aircc_archive/models` on local disk. Staging from there
removes the slow half of the sshfs-read → ssh-write hop: reading the mount measured ~3.4 MB/s, so a
2.8 GB model took ~14 minutes.

The mirror is only used when it is *provably* a faithful copy of what we are about to send, checked
at two levels:

1. **Global** — the last backup run in `backup.log` validated (reusing
   `daily_monitor.check_backup_log`, so there is one definition of "the backup succeeded").
2. **Per file** — every file we intend to stage exists in the mirror with the same size *and* mtime
   as the AIRCC source. `rsync -rt` preserves mtime, so a mismatch means the mirror is stale or the
   file was still in flight.

Either gate failing falls back to the AIRCC mount for that model. A model that finished after the
last backup simply is not in the mirror yet, and is handled by the same fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from aa_sweep import config


@dataclass
class SourceChoice:
    """Where to rsync one model's files from, and why."""

    path: Path
    label: str          # "mirror" or "aircc-mount"
    reason: str = ""    # why the mirror was rejected, when it was


def backup_log_ok(log_path: Path | None = None) -> tuple[bool, str]:
    """Did the most recent nightly backup complete cleanly?"""
    log_path = log_path or config.BACKUP_LOG
    try:
        from aircc.aircc_job_manager.daily_monitor import check_backup_log
    except Exception as exc:  # pragma: no cover - import guard only
        return False, f"cannot import backup log checker: {exc}"
    result = check_backup_log(log_path, 0)
    return result.ok, result.message


def _stat(path: Path) -> tuple[int, int] | None:
    """(size, mtime-to-the-second) or None. Whole seconds: rsync -rt does not preserve ns."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_size, int(st.st_mtime)


def files_match(mirror_dir: Path, source_dir: Path, filenames: list[str]) -> tuple[bool, str]:
    """Every named file must be byte-count and mtime identical in the mirror."""
    for filename in filenames:
        mirrored = _stat(mirror_dir / filename)
        if mirrored is None:
            return False, f"{filename} not in mirror"
        original = _stat(source_dir / filename)
        if original is None:
            return False, f"{filename} not readable on the aircc mount"
        if mirrored[0] != original[0]:
            return False, f"{filename} size {mirrored[0]} != aircc {original[0]}"
        if mirrored[1] != original[1]:
            return False, f"{filename} mtime differs from aircc (backup stale or mid-flight)"
    return True, "mirror matches aircc for every staged file"


def choose_source(
    model_name: str,
    aircc_dir: Path,
    filenames: list[str],
    backup_ok: bool,
    backup_message: str = "",
) -> SourceChoice:
    """Pick the mirror when it is verified to hold exactly these files, else the AIRCC mount.

    ``backup_ok`` is computed once per run by the caller rather than per model -- it is a property
    of the nightly backup, not of any one model.
    """
    if not config.USE_MIRROR:
        return SourceChoice(aircc_dir, "aircc-mount", "mirror disabled")
    if not backup_ok:
        return SourceChoice(aircc_dir, "aircc-mount", f"last backup did not validate: {backup_message}")

    mirror_dir = config.BACKUP_MIRROR / model_name
    if not mirror_dir.is_dir():
        return SourceChoice(aircc_dir, "aircc-mount", "model not in mirror yet")

    # The CSVs and selection json ride along with every staging rsync, so they have to be verified
    # too -- a stale CSV would hide cells the sweep should have run.
    small_files = [
        p.name for p in mirror_dir.iterdir()
        if p.is_file() and any(p.match(pattern) for pattern in config.STAGE_ALWAYS_GLOBS)
    ]
    ok, reason = files_match(mirror_dir, aircc_dir, sorted(set(filenames) | set(small_files)))
    if not ok:
        return SourceChoice(aircc_dir, "aircc-mount", reason)
    return SourceChoice(mirror_dir, "mirror")
