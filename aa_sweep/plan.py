"""Turn "what is finished in the two job DBs" into "what work each lane should run".

Reads both SQLite queues read-only -- the sjm one over the BGU sshfs mount, the AIRCC one from the
frozen snapshot in the QNAP archive -- then splits every finished model into exactly one of two
independent lanes:

* **Slurm** -- the model has a directory on the BGU cluster. The cluster attacks its own copy and
  writes its own CSVs; the planner's only job over there is a read-only ssh census.
* **Botero** -- the model was finished on AIRCC and the cluster does *not* have a directory for it.
  This machine attacks its own copy out of ``config.BOTERO_STORE_ROOT`` and writes its results
  beside it.

The split is by construction disjoint, which is what makes it safe for each lane's missing-cell
census to look only at its own machine's CSVs (see census.py). No model is ever copied between the
two: whatever propagation is wanted between machines is the weekly rsync's business.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from aa_sweep import botero, config
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


SLURM_LANE = config.SLURM_LANE
BOTERO_LANE = config.BOTERO_LANE


@dataclass
class ModelWork:
    """One finished model, assigned to exactly one lane, with that lane's own census."""

    model_name: str
    lane: str = SLURM_LANE
    sources: set[str] = field(default_factory=set)
    # The dir the lane's evaluation reads and writes. A remote path string for the Slurm lane, a
    # local Path for the Botero lane.
    slurm_dir: str = ""
    botero_dir: Path | None = None
    kinds: dict[str, KindStatus] = field(default_factory=dict)

    @property
    def runnable_kinds(self) -> list[str]:
        return [k for k in config.CHECKPOINT_KINDS if k in self.kinds and self.kinds[k].runnable]

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
    botero_reader=None,
) -> list[ModelWork]:
    """Assign every finished model to one lane and census it there.

    A model goes to the **Slurm** lane if the ssh probe found a directory for it on the cluster --
    whichever DB reported it finished. 102 of the 127 AIRCC-finished models are in this position,
    having been copied over by the staging step this package used to have; the cluster owns them now
    and this machine never touches them again.

    Everything else that AIRCC finished goes to the **Botero** lane, evaluated from this machine's
    own copy. Models finished only by sjm but with no cluster directory are dropped: the cluster
    cannot attack what it does not have, and they are not this machine's to run.

    ``botero_reader(model_name) -> (files, csvs, dir)`` is injectable so tests can supply fixtures;
    the default reads ``config.BOTERO_STORE_ROOT``. It is consulted only for Botero-lane candidates,
    so the local stat walk stays small.
    """
    if botero_reader is None:
        def botero_reader(name: str):
            model_dir = botero.resolve_model_dir(name)
            if model_dir is None:
                return {}, {}, None
            files, csvs = _read_local_dir(model_dir)
            return files, csvs, model_dir

    grid: set[Cell] = grid_cells(config.NORMS, config.EPS_INPUTS)

    sources: dict[str, set[str]] = {}
    for name in sjm_finished:
        sources.setdefault(name, set()).add("sjm")
    for name in aircc_finished:
        sources.setdefault(name, set()).add("aircc")

    plan: list[ModelWork] = []
    for model_name in sorted(sources):
        probe = slurm_probe.get(model_name, {})
        on_slurm = bool(probe.get("exists"))

        if on_slurm:
            files: dict[str, int] = probe.get("files", {}) or {}
            csvs: dict[str, str] = probe.get("csvs", {}) or {}
            work = ModelWork(
                model_name=model_name,
                lane=SLURM_LANE,
                sources=sources[model_name],
                slurm_dir=f"{config.SLURM_MODELS_ROOT}/{model_name}",
            )
        elif "aircc" in sources[model_name]:
            files_i, csvs, botero_dir = botero_reader(model_name)
            if botero_dir is None:
                # Finished on AIRCC, absent from the cluster, and no local copy either -- the three
                # `models_failed/` runs that never wrote a checkpoint land here. Nothing to run.
                continue
            files = files_i
            work = ModelWork(
                model_name=model_name,
                lane=BOTERO_LANE,
                sources=sources[model_name],
                botero_dir=botero_dir,
            )
        else:
            continue

        for kind in config.CHECKPOINT_KINDS:
            work.kinds[kind] = kind_status(
                kind=kind,
                ckpt_filename=config.CKPT_FILE_FOR_KIND[kind],
                # CSV rows key off the *directory* name, which is the last path segment for the
                # nested sjm names (vit_b_cvst/linf_1_init1 -> linf_1_init1).
                model_name=model_name.rsplit("/", 1)[-1],
                grid=grid,
                files=set(files),
                csv_text=csvs.get(kind, ""),
            )

        plan.append(work)
    return plan
