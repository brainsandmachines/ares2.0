"""Turn "what is finished in the two job DBs" into "what sbatch jobs should exist".

Reads both SQLite queues read-only over their sshfs mounts, then censuses each finished model on
both clusters before a single byte is copied, so models whose sweep is already complete cost
nothing at all.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from aa_sweep import config
from aa_sweep.census import Cell, KindStatus, grid_cells, kind_status

# Read the BGU side in one ssh round trip: list the run dir and slurp the three sweep CSVs for
# every candidate at once, rather than one ssh per model. Same "pipe a program over ssh stdin"
# technique the slurm-status skill uses.
_REMOTE_PROBE = r"""
import json, os, sys

payload = json.load(sys.stdin)
root = payload["root"]
out = {}
for name in payload["models"]:
    d = os.path.join(root, name)
    entry = {"exists": os.path.isdir(d), "files": {}, "csvs": {}}
    if entry["exists"]:
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                try:
                    entry["files"][fn] = os.path.getsize(p)
                except OSError:
                    entry["files"][fn] = -1
        for kind, csv_name in payload["csv_for_kind"].items():
            p = os.path.join(d, csv_name)
            if os.path.isfile(p):
                try:
                    with open(p, "r", newline="") as fh:
                        entry["csvs"][kind] = fh.read()
                except OSError:
                    pass
    out[name] = entry
json.dump(out, sys.stdout)
"""


@dataclass
class ModelWork:
    """Everything the submitter needs to know about one finished model."""

    model_name: str
    sources: set[str] = field(default_factory=set)
    slurm_dir: str = ""
    aircc_dir: Path | None = None
    slurm_dir_exists: bool = False
    kinds: dict[str, KindStatus] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    # Raw sweep-CSV text per kind from each side, kept so the stager can merge AIRCC-only rows
    # into an existing BGU CSV without re-reading anything.
    slurm_csvs: dict[str, str] = field(default_factory=dict)
    aircc_csvs: dict[str, str] = field(default_factory=dict)

    @property
    def runnable_kinds(self) -> list[str]:
        return [k for k in config.CHECKPOINT_KINDS if k in self.kinds and self.kinds[k].runnable]

    @property
    def staging_files(self) -> list[str]:
        """Checkpoint filenames to copy over: only for kinds that actually have missing cells."""
        return [
            config.CKPT_FILE_FOR_KIND[k]
            for k in config.CHECKPOINT_KINDS
            if k in self.kinds and self.kinds[k].needs_staging
        ]

    @property
    def is_complete(self) -> bool:
        return not self.runnable_kinds

    @property
    def missing_cell_count(self) -> int:
        return sum(len(s.missing) for s in self.kinds.values() if s.runnable)


def finished_models(db_path: Path) -> list[str]:
    """Model names with status='finished'. immutable=1 -- no locking/WAL writes over sshfs."""
    if not db_path.exists():
        raise FileNotFoundError(f"job DB not found (is the mount up?): {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    try:
        return [row[0] for row in conn.execute("SELECT model_name FROM jobs WHERE status='finished'")]
    finally:
        conn.close()


def _read_local_dir(model_dir: Path) -> tuple[dict[str, int], dict[str, str]]:
    """Files (name -> size) and sweep CSV text per kind, from a locally reachable model dir."""
    files: dict[str, int] = {}
    csvs: dict[str, str] = {}
    if not model_dir.is_dir():
        return files, csvs
    for path in model_dir.iterdir():
        if path.is_file():
            try:
                files[path.name] = path.stat().st_size
            except OSError:
                files[path.name] = -1
    for kind, csv_name in config.CSV_FOR_KIND.items():
        path = model_dir / csv_name
        if path.is_file():
            try:
                with path.open("r", newline="") as fh:
                    csvs[kind] = fh.read()
            except OSError:
                pass
    return files, csvs


def probe_slurm(model_names: list[str], run=subprocess.run) -> dict[str, dict]:
    """One batched ssh round trip returning each model dir's files+CSVs on the BGU cluster."""
    if not model_names:
        return {}
    payload = json.dumps(
        {
            "root": config.SLURM_MODELS_ROOT,
            "models": model_names,
            "csv_for_kind": config.CSV_FOR_KIND,
        }
    )
    # stdin carries the JSON payload, so the program itself rides along base64-encoded inside
    # -c. That sidesteps quoting it through ssh's remote shell entirely.
    encoded = base64.b64encode(_REMOTE_PROBE.encode()).decode()
    remote = f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    proc = run(
        ["ssh", "-o", f"ConnectTimeout={config.SSH_TIMEOUT_SECONDS}", config.SLURM_SSH_HOST, remote],
        input=payload,
        capture_output=True,
        text=True,
        timeout=config.SSH_TIMEOUT_SECONDS * 4,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh probe of {config.SLURM_SSH_HOST} failed rc={proc.returncode}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def build_plan(
    aircc_finished: list[str],
    sjm_finished: list[str],
    slurm_probe: dict[str, dict],
    aircc_reader=None,
) -> list[ModelWork]:
    """Census every finished model on both clusters and return one ModelWork each.

    ``aircc_reader(model_name) -> (files, csvs)`` is injectable so tests can supply fixtures
    instead of an sshfs mount; the default reads the real AIRCC mount.
    """
    if aircc_reader is None:
        def aircc_reader(name: str):
            return _read_local_dir(config.AIRCC_MODELS_ROOT / name)

    grid: set[Cell] = grid_cells(config.NORMS, config.EPS_INPUTS)

    sources: dict[str, set[str]] = {}
    for name in sjm_finished:
        sources.setdefault(name, set()).add("sjm")
    for name in aircc_finished:
        sources.setdefault(name, set()).add("aircc")

    plan: list[ModelWork] = []
    for model_name in sorted(sources):
        probe = slurm_probe.get(model_name, {})
        slurm_files: dict[str, int] = probe.get("files", {}) or {}
        slurm_csvs: dict[str, str] = probe.get("csvs", {}) or {}

        is_aircc = "aircc" in sources[model_name]
        aircc_files: dict[str, int] = {}
        aircc_csvs: dict[str, str] = {}
        if is_aircc:
            aircc_files, aircc_csvs = aircc_reader(model_name)

        work = ModelWork(
            model_name=model_name,
            sources=sources[model_name],
            slurm_dir=f"{config.SLURM_MODELS_ROOT}/{model_name}",
            aircc_dir=(config.AIRCC_MODELS_ROOT / model_name) if is_aircc else None,
            slurm_dir_exists=bool(probe.get("exists")),
            slurm_csvs=slurm_csvs,
            aircc_csvs=aircc_csvs,
        )

        for kind in config.CHECKPOINT_KINDS:
            ckpt = config.CKPT_FILE_FOR_KIND[kind]
            work.kinds[kind] = kind_status(
                kind=kind,
                ckpt_filename=ckpt,
                # CSV rows key off the *directory* name, which is the last path segment for the
                # nested sjm names (vit_b_cvst/linf_1_init1 -> linf_1_init1).
                model_name=model_name.rsplit("/", 1)[-1],
                grid=grid,
                slurm_files=set(slurm_files),
                slurm_csv_text=slurm_csvs.get(kind, ""),
                aircc_files=set(aircc_files),
                aircc_csv_text=aircc_csvs.get(kind, ""),
            )
            # Both sides hold this checkpoint at different sizes: two different training runs
            # sharing a name across clusters. Never overwrite the BGU one -- report and move on.
            if ckpt in slurm_files and ckpt in aircc_files and slurm_files[ckpt] != aircc_files[ckpt]:
                work.conflicts.append(
                    f"{ckpt}: slurm={slurm_files[ckpt]}B aircc={aircc_files[ckpt]}B (keeping slurm)"
                )

        plan.append(work)
    return plan
