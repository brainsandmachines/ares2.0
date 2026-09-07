"""Enumerate every trained model across all four archive roots and both job DBs.

This is the shared substrate for every other pass: it answers "what models exist,
where do their files live, and how does each decompose into arch/protocol/norm/eps".

Decomposition precedence, most authoritative first:

1. **The job-manager CSVs** -- ``arch, protocol, threat_norm, threat_eps, init`` are
   explicit columns, joined on ``model_name``. Covers convnext_base (AIRCC) and
   vit_b_cvst / swin_b (Slurm): 318/323 AIRCC rows and 62/62 Slurm rows.
2. **The model's own config** -- ``model_store.config_reader``. Covers the 174
   convnext_small dirs (in no CSV) and the five hand-launched ``*_pgd5*`` runs.
3. **The folder name** -- cross-check only, never a source of truth on its own.

Anything that survives none of the three is reported and routed to
``models/_legacy/unparsed/`` rather than filed on a guess.

Read-only. Opens both DBs ``?immutable=1``, as every other reader in this repo does.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

from . import config_reader
from .naming import (
    LEGACY_CONTAINERS, ModelIdentity, arch_from_name,
    canonical_name, is_intermediate, is_keeper_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- roots -----------------------------------------------------------------
# QNAP first: it is the master. /mnt/data4t second. Order is the search order
# used when resolving a checkpoint basename to a real file.
QNAP_ROOT = Path(os.environ.get("MS_QNAP_ROOT", "/mnt/botero"))
DATA4T_ROOT = Path(os.environ.get("MS_DATA4T_ROOT", "/mnt/data4t"))
STORE_ROOT = Path(os.environ.get("MS_STORE_ROOT", "/mnt/data4t/models"))
EXPERIMENTS_ROOT = Path(os.environ.get("MS_EXPERIMENTS_ROOT", "/mnt/data4t/models_for_experiments"))

ARCHIVE_ROOTS = {
    "qnap-slurm": QNAP_ROOT / "slurm_archive" / "models",
    "qnap-aircc": QNAP_ROOT / "aircc_archive" / "models",
    "data4t-slurm": DATA4T_ROOT / "slurm_archive" / "models",
    "data4t-aircc": DATA4T_ROOT / "aircc_archive" / "models",
}

# --- databases -------------------------------------------------------------
# The live Slurm queue over sshfs: still growing (11 running / 11 pending as of
# 2026-09-02), so it must be read fresh, not from a snapshot.
SJM_DB = Path(os.environ.get(
    "MS_SJM_DB", Path.home() / "slurm_mount/projects/ares/slurm_job_manager/jobs.sqlite"))
# The AIRCC campaign is finished and its cluster tree deleted; this snapshot on
# local disk was row-diffed identical to the last live DB and is now the record.
#
# It lives under the curated tree, not in the old ``aircc_archive/`` layout that
# this package replaces: the archive root is staged for deletion once the user
# erases pending_deletion/, and a zoo that silently lost every AIRCC blessing
# would still build -- just with SJM rows only. ``models/_meta/`` is inside the
# thing the weekly backup keeps, so the record travels with the models it
# describes. The QNAP keeps its own copy either way.
AIRCC_DB = Path(os.environ.get(
    "MS_AIRCC_DB",
    STORE_ROOT / "_meta" / "aircc_archive" / "aircc_jobs_final_latest.sqlite"))


@dataclass
class ModelRecord:
    """One model, everything known about it."""

    identity: ModelIdentity
    # root label -> the model dir under that root
    dirs: dict[str, Path] = field(default_factory=dict)
    # DB facts, when the model has a row
    db_source: Optional[str] = None       # "sjm" | "aircc"
    db_model_name: Optional[str] = None   # verbatim, may contain "/"
    db_status: Optional[str] = None
    best_checkpoint: Optional[str] = None  # verbatim cluster path from the DB
    best_score: Optional[float] = None
    # Set when >1 on-disk dir canonicalises to this record's key -- never merged
    # silently, always reported.
    collision_names: list[str] = field(default_factory=list)

    @property
    def is_trained(self) -> bool:
        """Has this model actually produced a checkpoint?

        A DB row alone does not mean trained: 190 of the 385 rows are ``pending``
        (planned but never claimed) and 27 are ``running``. Only a dir on disk, or
        a finished row, is evidence of a real model.
        """
        return bool(self.dirs) or self.db_status == "finished"

    @property
    def best_basename(self) -> Optional[str]:
        """The only portable part of ``best_checkpoint``.

        The column holds three different cluster roots, two of which no longer
        exist (``/home/ashtomer/projects/ares/results``,
        ``/groups/golan_neurogroup/.../advmodels/results``,
        ``/shared/cycle2_bgu_golan_prj/.../ares/results``), so the path is only
        usable as a basename to be re-rooted locally.
        """
        if not self.best_checkpoint:
            return None
        return _posix_basename(self.best_checkpoint)


def _posix_basename(path: str) -> str:
    """Basename of a POSIX cluster path, independent of the local separator."""
    return path.rstrip("/").rsplit("/", 1)[-1]


# --------------------------------------------------------------------------
# CSV metadata
# --------------------------------------------------------------------------
def load_csv_metadata() -> dict[str, dict]:
    """``model_name`` -> CSV row, unioned across both managers' CSV dirs.

    Keyed on the DB's ``model_name`` verbatim (so ``swin_b/l2_4_init1``), because
    that is the join key both managers use.
    """
    rows: dict[str, dict] = {}
    for pkg in ("slurm_job_manager", "aircc.aircc_job_manager"):
        try:
            mod = __import__(f"{pkg}.csv_spec", fromlist=["csv_spec"])
        except ImportError:
            continue
        try:
            for row in mod.load_all_rows():
                name = (row.get("model_name") or "").strip()
                if name:
                    rows[name] = row
        except Exception:
            continue
    return rows


def identity_from_csv(name: str, row: dict) -> ModelIdentity:
    canonical = canonical_name(name)
    arch = (row.get("arch") or "").strip() or arch_from_name(canonical)
    protocol = (row.get("protocol") or "").strip() or None
    norm = (row.get("threat_norm") or "").strip().lower() or None
    eps_raw = (row.get("threat_eps") or "").strip()
    try:
        eps = float(eps_raw) if eps_raw else None
    except ValueError:
        eps = None
    return ModelIdentity(
        canonical=canonical, arch=arch, protocol=protocol, norm=norm, eps=eps,
        init=(row.get("init") or "").strip() or None, source="csv",
        legacy=False,
    )


# --------------------------------------------------------------------------
# Databases
# --------------------------------------------------------------------------
def _read_db(db_path: Path, label: str) -> list[dict]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT model_name, status, best_checkpoint, best_score FROM jobs")
        return [dict(r, _source=label) for r in cur.fetchall()]
    finally:
        con.close()


def load_db_rows() -> list[dict]:
    return _read_db(SJM_DB, "sjm") + _read_db(AIRCC_DB, "aircc")


# --------------------------------------------------------------------------
# Disk walk
# --------------------------------------------------------------------------
def _looks_like_model_dir(path: Path) -> bool:
    """A model dir is one that holds an actual checkpoint.

    Deliberately **not** "holds a config": ``old_models/`` itself carries a stray
    ``args.yaml`` and ``log.txt`` left over from an old run, so a config-based test
    classifies that container as a model and stops the recursion before reaching
    the 22 real models under ``old_models/{madry,gradnorm}/``. A checkpoint is the
    only unambiguous marker, and a dir with configs but no weights is not a model
    worth storing anyway.
    """
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                if is_keeper_checkpoint(entry.name) or is_intermediate(entry.name):
                    return True
            elif entry.is_dir(follow_symlinks=False) and entry.name == "periodic":
                # A periodic/ dir only ever exists inside a model dir.
                return True
    except OSError:
        return False
    return False


# How far below ``<archive>/models`` a model dir can sit. 1 = flat AIRCC/convnext,
# 2 = the nested ViT/Swin lanes and most of old_models, 3 = old_models/madry/<model>
# and old_models/gradnorm/<model> (22 dirs, 44 checkpoints, which a 2-deep walk
# silently skipped).
MAX_MODEL_DEPTH = 3


def walk_root(root: Path, max_depth: int = MAX_MODEL_DEPTH) -> dict[str, Path]:
    """``model_name`` (relative to ``root``, ``/``-joined) -> dir.

    Handles every convention in one bounded recursive pass: flat
    ``<root>/<name>``, nested ``<root>/<arch>/<name>`` for the ViT/Swin lanes, and
    the deeper ``<root>/old_models/<protocol>/<name>`` legacy layout. Recursion
    stops as soon as a dir looks like a model, so a model's own ``periodic/`` or
    ``pgd_eval/`` subdir is never mistaken for another model.
    """
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found

    def descend(path: Path, rel: str, depth: int) -> None:
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            sub = Path(entry.path)
            sub_rel = f"{rel}/{entry.name}" if rel else entry.name
            if _looks_like_model_dir(sub):
                found[sub_rel] = sub
            elif depth < max_depth:
                descend(sub, sub_rel, depth + 1)

    descend(root, "", 1)
    return found


# --------------------------------------------------------------------------
# The census itself
# --------------------------------------------------------------------------
def build(roots: Optional[Iterable[str]] = None) -> dict[str, ModelRecord]:
    """Canonical name -> :class:`ModelRecord`, merged across roots and DBs."""
    labels = list(roots) if roots else list(ARCHIVE_ROOTS)
    csv_meta = load_csv_metadata()
    db_rows = load_db_rows()

    # A CSV row is authoritative about a *run* only when a DB row proves that CSV
    # launched it. ``aircc/.../csv/convnext_small.csv`` is a plan that was never
    # run there, yet 40 of its 312 rows name-collide with old Slurm convnext_small
    # dirs -- so for a model with no DB row the dir's own config wins over the CSV.
    db_canonical = {canonical_name((r.get("model_name") or "").strip())
                    for r in db_rows if (r.get("model_name") or "").strip()}

    # 1. disk
    per_label: dict[str, dict[str, Path]] = {}
    for label in labels:
        per_label[label] = walk_root(ARCHIVE_ROOTS[label])

    records: dict[str, ModelRecord] = {}
    collisions: dict[str, set[str]] = defaultdict(set)

    for label, mapping in per_label.items():
        for raw_name, path in mapping.items():
            canonical = canonical_name(raw_name)
            container = raw_name.split("/")[0] if "/" in raw_name else ""
            ident = _decompose(
                raw_name, canonical, path, csv_meta, container,
                csv_authoritative=canonical in db_canonical,
            )
            # Legacy models key on their full archive path, so a legacy dir can
            # neither shadow a live model of the same canonical name nor collide
            # with a sibling under a different protocol subdir.
            key = ident.record_key
            collisions[key].add(raw_name)
            rec = records.get(key)
            if rec is None:
                rec = ModelRecord(identity=ident)
                records[key] = rec
            elif label in rec.dirs and rec.dirs[label] != path:
                # Two distinct dirs under one root claiming one identity: merging
                # them would silently drop a model, so surface it instead.
                rec.identity = replace(
                    rec.identity, legacy=True, notes="collision",
                )
            rec.dirs[label] = path

    for key, raws in collisions.items():
        if len(raws) > 1 and key in records:
            records[key].collision_names = sorted(raws)

    # 2. DB rows -- may name a model that has no dir anywhere (the 3 AIRCC orphans)
    for row in db_rows:
        name = (row.get("model_name") or "").strip()
        if not name:
            continue
        canonical = canonical_name(name)
        rec = records.get(canonical)
        if rec is None:
            ident = _decompose(name, canonical, None, csv_meta, "", csv_authoritative=True)
            rec = ModelRecord(identity=ident)
            records[canonical] = rec
        rec.db_source = row["_source"]
        rec.db_model_name = name
        rec.db_status = row.get("status")
        rec.best_checkpoint = row.get("best_checkpoint")
        rec.best_score = row.get("best_score")

    return records


def _decompose(
    raw_name: str, canonical: str, model_dir: Optional[Path],
    csv_meta: dict[str, dict], container: str, csv_authoritative: bool = True,
) -> ModelIdentity:
    """CSV -> config -> name, with the legacy containers short-circuited.

    ``csv_authoritative`` is False for a dir with no DB row: then the dir's own
    config is tried first and the CSV is only a fallback.
    """
    if container in LEGACY_CONTAINERS:
        ident = None
        if model_dir is not None:
            ident = config_reader.read_identity(model_dir, fallback_name=canonical)
        return ModelIdentity(
            canonical=canonical,
            arch=(ident.arch if ident else arch_from_name(canonical)),
            protocol=(ident.protocol if ident else None),
            norm=(ident.norm if ident else None),
            eps=(ident.eps if ident else None),
            init=(ident.init if ident else None),
            source=(ident.source if ident else "name"),
            legacy=True, notes=container, legacy_relpath=raw_name,
        )

    row = csv_meta.get(raw_name) or csv_meta.get(canonical)
    if row and csv_authoritative:
        return identity_from_csv(raw_name, row)

    if model_dir is not None:
        ident = config_reader.read_identity(model_dir, fallback_name=canonical)
        if ident is not None and not ident.legacy:
            return ident

    if row:
        return identity_from_csv(raw_name, row)

    if model_dir is not None:
        ident = config_reader.read_identity(model_dir, fallback_name=canonical)
        if ident is not None:
            return ident

    return ModelIdentity(
        canonical=canonical, arch=arch_from_name(canonical), protocol=None,
        norm=None, eps=None, init=None, source="name",
        legacy=True, notes="unparsed",
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cmd_explain(args) -> int:
    records = build(roots=args.roots or None)
    key = canonical_name(args.explain)
    rec = records.get(key)
    if rec is None:  # maybe a legacy-namespaced key
        matches = [k for k in records if k.endswith("/" + key)]
        rec = records[matches[0]] if len(matches) == 1 else None
    if rec is None:
        print(f"no such model: {args.explain}", file=sys.stderr)
        return 1
    i = rec.identity
    print(f"canonical    : {i.canonical}")
    print(f"arch         : {i.arch}")
    print(f"protocol     : {i.protocol}")
    print(f"norm / eps   : {i.norm} / {i.eps}")
    print(f"init         : {i.init}")
    print(f"decomposed by: {i.source}")
    print(f"legacy       : {i.legacy}{(' (' + i.notes + ')') if i.notes else ''}")
    print(f"store path   : models/{i.store_relpath}")
    print(f"experiment   : {i.experiment_relpath or '(none)'}")
    print(f"db           : {rec.db_source} status={rec.db_status} score={rec.best_score}")
    print(f"db best ckpt : {rec.best_checkpoint}")
    print(f"  -> basename: {rec.best_basename}")
    for label, path in sorted(rec.dirs.items()):
        print(f"dir[{label:<13}] {path}")
    return 0


def _cmd_report_unparsed(args) -> int:
    records = build(roots=args.roots or None)
    legacy = sorted(
        (r for r in records.values() if r.identity.legacy),
        key=lambda r: (r.identity.notes, r.identity.canonical),
    )
    by_bucket: dict[str, list[ModelRecord]] = defaultdict(list)
    for rec in legacy:
        by_bucket[rec.identity.notes or "unparsed"].append(rec)
    for bucket, recs in sorted(by_bucket.items()):
        print(f"\n=== _legacy/{bucket}  ({len(recs)} models) ===")
        for rec in recs:
            i = rec.identity
            where = ",".join(sorted(rec.dirs)) or "NO DIR"
            print(f"  {i.canonical:<62} arch={i.arch or '-':<15} src={i.source:<28} [{where}]")
    print(f"\ntotal legacy/unparsed: {len(legacy)} of {len(records)} models")
    return 0


def _cmd_summary(args) -> int:
    records = build(roots=args.roots or None)
    trained = {k: r for k, r in records.items() if r.is_trained}
    print(f"records total : {len(records)}")
    print(f"trained models: {len(trained)}   (a dir on disk, or a finished DB row)")
    print(f"planned only  : {len(records) - len(trained)}   (pending/running rows, no dir)")

    by_arch: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    by_proto: dict[str, int] = defaultdict(int)
    for rec in trained.values():
        by_arch[rec.identity.arch or "(none)"] += 1
        by_source[rec.identity.source.split(":")[0]] += 1
        by_proto[rec.identity.protocol or "(none)"] += 1
    for title, data in (("arch", by_arch), ("decomposed by", by_source), ("protocol", by_proto)):
        print(f"\ntrained models by {title}:")
        for key, count in sorted(data.items(), key=lambda kv: -kv[1]):
            print(f"  {count:5d}  {key}")

    # The real gaps: finished, blessed by the DB, but no directory anywhere.
    gaps = [r for r in records.values()
            if r.db_status == "finished" and r.best_checkpoint and not r.dirs]
    print(f"\nfinished + DB best_checkpoint but NO dir on any root: {len(gaps)}")
    for rec in sorted(gaps, key=lambda r: r.identity.canonical):
        print(f"  GAP {rec.identity.canonical:<52} db={rec.db_source} "
              f"score={rec.best_score} best={rec.best_basename}")

    colliding = [r for r in records.values() if r.collision_names]
    if colliding:
        print(f"\ncanonical-name collisions: {len(colliding)}")
        for rec in colliding:
            print(f"  {rec.identity.canonical}: {rec.collision_names}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--explain", metavar="MODEL", help="show how one model decomposes")
    ap.add_argument("--report-unparsed", action="store_true",
                    help="list every model routed to models/_legacy/")
    ap.add_argument("--summary", action="store_true", help="counts by arch/protocol/source")
    ap.add_argument("--roots", nargs="*", choices=sorted(ARCHIVE_ROOTS),
                    help="limit the disk walk (default: all four)")
    args = ap.parse_args(argv)

    if args.explain:
        return _cmd_explain(args)
    if args.report_unparsed:
        return _cmd_report_unparsed(args)
    return _cmd_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
