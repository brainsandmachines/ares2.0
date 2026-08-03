"""Copy an AIRCC model onto the BGU cluster, carefully.

Two rules, both load-bearing:

1. **Only the gaps.** Checkpoints are ~1.4 GB each; a model that needs only its ``advbest`` kind
   swept transfers one file, not three. Models with no missing cells never reach this module.
2. **Never overwrite.** ``--ignore-existing`` means the rsync can only *add* files. Ten AIRCC model
   names already exist as directories on the BGU cluster, and five of those are real BGU-trained
   runs whose AIRCC counterpart is a near-empty husk -- copying over them would destroy real
   weights.

Rule 2 has one gap: when both sides hold a sweep CSV for the same kind, ``--ignore-existing``
keeps the BGU file and would silently drop AIRCC-only rows. Those rows are exactly the eps_norm
results we promised to reuse, so for that case we merge instead, via the engine's own
``write_rows`` upsert.
"""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from aa_sweep import config


@dataclass
class StageResult:
    model_name: str
    files: list[str] = field(default_factory=list)
    bytes_planned: int = 0
    merged_csvs: list[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""


def rsync_command(model_name: str, aircc_dir: Path, checkpoint_files: list[str]) -> list[str]:
    """Build the staging rsync: the always-carried small files plus the named checkpoints only."""
    includes: list[str] = []
    for pattern in config.STAGE_ALWAYS_GLOBS:
        includes += ["--include", pattern]
    for filename in checkpoint_files:
        includes += ["--include", filename]
    dest = f"{config.SLURM_SSH_HOST}:{config.SLURM_MODELS_ROOT}/{model_name}/"
    return [
        "rsync",
        "-rt",
        "--ignore-existing",
        "--no-perms",
        "--no-owner",
        "--no-group",
        "--partial",
        *includes,
        # Everything not explicitly included above is skipped -- notably the rolling
        # checkpoint-N.pth.tar files (~1.4GB each) and the pgd_eval/ subtree.
        "--exclude",
        "*",
        f"{aircc_dir}/",
        dest,
    ]


def merge_csv_rows(aircc_text: str, slurm_text: str) -> str | None:
    """Upsert AIRCC-only (norm, eps) rows into the BGU CSV. None when nothing would change.

    Keyed on (attack_norm, epsilon_input) exactly like
    ``data_analysis.autoattack_array_eval.write_rows``, and the BGU row always wins a tie.
    """
    if not aircc_text or not slurm_text:
        return None

    slurm_reader = csv.DictReader(io.StringIO(slurm_text))
    slurm_rows = list(slurm_reader)
    fieldnames = list(slurm_reader.fieldnames or [])
    if not fieldnames:
        return None

    def key(row: dict) -> tuple[str, float] | None:
        norm = (row.get("attack_norm") or "").strip().lower()
        eps = row.get("epsilon_input")
        if not norm or eps in (None, ""):
            return None
        try:
            return norm, round(float(eps), 10)
        except (TypeError, ValueError):
            return None

    have = {k for k in (key(row) for row in slurm_rows) if k is not None}
    extra = []
    for row in csv.DictReader(io.StringIO(aircc_text)):
        k = key(row)
        if k is not None and k not in have:
            have.add(k)
            extra.append(row)
    if not extra:
        return None

    for row in extra:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(slurm_rows + extra)
    return buf.getvalue()


def stage_model(work, aircc_csvs: dict[str, str], slurm_csvs: dict[str, str], run=subprocess.run,
                dry_run: bool = False) -> StageResult:
    """Stage one model's missing checkpoints, then merge any AIRCC-only CSV rows across."""
    result = StageResult(model_name=work.model_name)
    if work.aircc_dir is None:
        return result

    checkpoint_files = work.staging_files
    csvs_to_merge = {
        kind: merged
        for kind in config.CHECKPOINT_KINDS
        if (merged := merge_csv_rows(aircc_csvs.get(kind, ""), slurm_csvs.get(kind, ""))) is not None
    }
    if not checkpoint_files and not csvs_to_merge:
        return result

    result.files = checkpoint_files
    for filename in checkpoint_files:
        try:
            result.bytes_planned += (work.aircc_dir / filename).stat().st_size
        except OSError:
            pass

    if dry_run:
        result.merged_csvs = sorted(csvs_to_merge)
        return result

    cmd = rsync_command(work.model_name, work.aircc_dir, checkpoint_files)
    proc = run(cmd, capture_output=True, text=True, timeout=config.RSYNC_TIMEOUT_SECONDS)
    # 23/24 are the "file vanished / partial attrs" codes a live tree produces; the backup cron
    # forgives them the same way.
    if proc.returncode not in (0, 23, 24):
        result.ok = False
        result.error = f"rsync rc={proc.returncode}: {proc.stderr.strip()[:500]}"
        return result

    for kind, merged in csvs_to_merge.items():
        if _push_text(f"{config.SLURM_MODELS_ROOT}/{work.model_name}/{config.CSV_FOR_KIND[kind]}",
                      merged, run=run):
            result.merged_csvs.append(kind)
        else:
            result.ok = False
            result.error = f"failed writing merged {config.CSV_FOR_KIND[kind]}"
    return result


def _push_text(remote_path: str, text: str, run=subprocess.run) -> bool:
    """Write text to a file on the cluster over ssh (small CSVs only)."""
    proc = run(
        ["ssh", "-o", f"ConnectTimeout={config.SSH_TIMEOUT_SECONDS}", config.SLURM_SSH_HOST,
         f"cat > {remote_path}"],
        input=text,
        capture_output=True,
        text=True,
        timeout=config.SSH_TIMEOUT_SECONDS * 2,
    )
    return proc.returncode == 0
