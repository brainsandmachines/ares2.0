"""Step 6: build ``models_for_experiments`` -- the symlink zoo.

    models_for_experiments/
      convnext_base/madry/l2/convnext_base_l2_4_init1.pth.tar   -> the blessed checkpoint
      convnext_base/baseline/convnext_base_baseline_init0.pth.tar
      swin_b/madry/l1/swin_b_l1_2_init1.pth.tar
      vit_b_cvst/trades/l2/vit_b_cvst_l2trades_2_init1.pth.tar
      manifest.csv

The filename always carries the arch. That is not cosmetic: the ``swin_b`` and
``vit_b_cvst`` dirs name themselves ``l2_4_init1`` with the arch only in the parent,
and all 31 ``swin_b`` leaf names are also ``vit_b_cvst`` leaf names -- so a flat
zoo keyed on the leaf would silently collapse the two lanes.

**Only DB-blessed models get an entry.** A model appears here when a job DB
recorded a ``best_checkpoint`` for it -- i.e. a finished run whose winning kind was
scored at its trained threat model. That is the whole of convnext_base, swin_b and
vit_b_cvst. Everything else (convnext_small, vit_m_cvst, the legacy buckets) stays
in ``/mnt/data4t/models`` and is listed in the gaps report, but gets no symlink:
this tree is the *blessed* set, not an inventory.

**Which checkpoint gets linked**, in precedence order:

0. ``model_store/blessing_overrides.csv`` -- a hand correction for a model whose DB
   record is wrong: a different ``checkpoint``, and/or a ``publish_as`` name (the entry
   is published under that name, and the model that already carried it is dropped).
   It is checked in, so every rebuild re-applies it instead of reverting to the DB;
   each row must say why, and that reason lands in the manifest. Separately,
   ``model_store/zoo_excludes.csv`` lists relpath globs that are never published.
1. ``jobs.best_checkpoint`` from the job DB -- but only its **basename**. The column
   stores three different cluster roots, two of which no longer exist, so the path
   itself is dead. The basename is one of ``last`` / ``model_best`` /
   ``model_best_adv``, and it is the DB's record of the kind that scored highest at
   the model's trained (norm, eps).
2. ``best_checkpoint_for_threat()`` over the model's AutoAttack sweep CSVs
   (``aircc/aircc_job_manager/best_checkpoint.py``) -- only reachable with
   ``--allow-unblessed``, for a DB row whose ``best_checkpoint`` is NULL.
3. ``model_best.pth.tar``, likewise, and recorded as a fallback so it is never
   mistaken for a scored decision.

Nothing is guessed silently: every entry's rule is written to ``manifest.csv``, and
models that resolve to nothing land in the gaps report instead of the tree.
"""

from __future__ import annotations

import argparse
import csv as _csv
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from .census import EXPERIMENTS_ROOT, SJM_DB, STORE_ROOT, ModelRecord, build
from .dedupe_report import LOG_DIR, _now
from .naming import CKPT_FILE_FOR_KIND, KIND_FOR_CKPT_FILE

GIB = 1024 ** 3
OVERRIDES_CSV = Path(__file__).with_name("blessing_overrides.csv")
EXCLUDES_CSV = Path(__file__).with_name("zoo_excludes.csv")


@dataclass
class Entry:
    canonical: str
    relpath: str            # <arch>/<protocol>/<norm>/<canonical>.pth.tar
    target: Path            # absolute path under models/
    kind: str               # best | last | advbest
    rule: str               # db | aa-sweep | fallback
    db_source: str = ""
    best_score: Optional[float] = None
    # Why a non-`db` rule fired, when one did. Recorded so a fallback is never
    # mistaken for a scored decision.
    note: str = ""
    # The model this entry was built from, when a publish_as override renamed it.
    source: str = ""


@dataclass
class Gap:
    canonical: str
    why: str
    detail: str = ""


@dataclass
class Override:
    checkpoint: Optional[str]   # a keeper basename, e.g. model_best_adv.pth.tar; None = DB decides
    publish_as: Optional[str]   # publish under this name, displacing the model that has it
    score: Optional[float]      # the winning kind's score, replacing the DB's
    reason: str


def load_overrides(path: Path) -> dict[str, Override]:
    """``canonical`` -> :class:`Override`, from the checked-in overrides CSV.

    Fails loudly on a malformed row: a typo here would otherwise demote the model
    back to its DB record on the next rebuild without anyone noticing.
    """
    if not path.exists():
        return {}
    out: dict[str, Override] = {}
    with path.open(newline="") as fh:
        for n, row in enumerate(_csv.DictReader(fh), start=2):
            model = (row.get("model") or "").strip()
            if not model:
                continue
            ckpt = (row.get("checkpoint") or "").strip() or None
            publish_as = (row.get("publish_as") or "").strip() or None
            reason = (row.get("reason") or "").strip()
            if ckpt is None and publish_as is None:
                raise ValueError(f"{path}:{n}: override for {model} sets neither "
                                 f"checkpoint nor publish_as")
            if ckpt is not None and ckpt not in KIND_FOR_CKPT_FILE:
                raise ValueError(f"{path}:{n}: checkpoint {ckpt!r} is not one of "
                                 f"{sorted(KIND_FOR_CKPT_FILE)}")
            if publish_as == model:
                raise ValueError(f"{path}:{n}: {model} is published as itself")
            if not reason:
                raise ValueError(f"{path}:{n}: override for {model} has no reason")
            if model in out:
                raise ValueError(f"{path}:{n}: duplicate override for {model}")
            score_raw = (row.get("score") or "").strip()
            out[model] = Override(ckpt, publish_as,
                                  float(score_raw) if score_raw else None, reason)
    targets = [o.publish_as for o in out.values() if o.publish_as]
    if len(targets) != len(set(targets)):
        raise ValueError(f"{path}: two overrides publish under the same name")
    # A chain (A as B, B as C) would make who displaces whom depend on row order.
    chained = set(targets) & {m for m, o in out.items() if o.publish_as}
    if chained:
        raise ValueError(f"{path}: publish_as chains through {sorted(chained)}")
    return out


def load_excludes(path: Path) -> dict[str, str]:
    """relpath glob -> reason, from the checked-in excludes CSV.

    Globs are :func:`fnmatch.fnmatchcase`, so ``*`` also crosses ``/``: a bare
    ``*_resetepoch.pth.tar`` applies to every arch, protocol and norm.
    """
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for n, row in enumerate(_csv.DictReader(fh), start=2):
            pattern = (row.get("pattern") or "").strip()
            if not pattern:
                continue
            reason = (row.get("reason") or "").strip()
            if not reason:
                raise ValueError(f"{path}:{n}: exclude {pattern!r} has no reason")
            if pattern in out:
                raise ValueError(f"{path}:{n}: duplicate exclude {pattern!r}")
            out[pattern] = reason
    return out


def _aa_sweep_kind(model_dir: Path, norm, eps) -> Optional[str]:
    """The kind the AutoAttack sweep CSVs would pick, or None."""
    if not model_dir.is_dir():
        return None
    from aircc.aircc_job_manager.best_checkpoint import best_checkpoint_for_threat
    try:
        path, _score = best_checkpoint_for_threat(model_dir, norm, eps)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not path:
        return None
    return KIND_FOR_CKPT_FILE.get(os.path.basename(path))


def _resolve_kind(
    rec: ModelRecord, store_root: Path, override: Optional[Override] = None,
) -> tuple[Optional[str], str, str, str]:
    """Return ``(checkpoint basename, kind, rule, note)`` or ``(None, "", reason, "")``."""
    model_dir = store_root / rec.identity.store_relpath

    # 0. a checked-in correction beats the DB record it corrects
    if override is not None and override.checkpoint:
        return (override.checkpoint, KIND_FOR_CKPT_FILE[override.checkpoint],
                "override", override.reason)

    # 1. the DB's own record
    basename = rec.best_basename
    if basename in CKPT_FILE_FOR_KIND.values():
        return basename, KIND_FOR_CKPT_FILE[basename], "db", ""

    # 2. score it from the AutoAttack sweep results still in the model dir.
    # ImportError inside _aa_sweep_kind is NOT swallowed: best_checkpoint_for_threat
    # needs pandas, and a missing dep would otherwise silently demote every
    # remaining model to the rule-4 fallback while looking like a clean run.
    aa = _aa_sweep_kind(model_dir, rec.identity.norm, rec.identity.eps)
    if aa:
        return CKPT_FILE_FOR_KIND[aa], aa, "aa-sweep", ""

    # 3. last resort, and labelled as such
    if (model_dir / "model_best.pth.tar").exists():
        return "model_best.pth.tar", "best", "fallback", "no AA scores for this model"
    if (model_dir / "last.pth.tar").exists():
        return "last.pth.tar", "last", "fallback", "no AA scores, no model_best"
    return None, "", "no checkpoint in the curated tree", ""


def plan(
    records: dict[str, ModelRecord], store_root: Path,
    allow_unblessed: bool = False,
    overrides: Optional[dict[str, Override]] = None,
    excludes: Optional[dict[str, str]] = None,
) -> tuple[list[Entry], list[Gap]]:
    entries: list[Entry] = []
    gaps: list[Gap] = []
    by_relpath: dict[str, list[str]] = defaultdict(list)
    overrides = overrides or {}
    excludes = excludes or {}
    used: set[str] = set()
    used_excludes: set[str] = set()

    for rec in sorted(records.values(), key=lambda r: r.identity.canonical):
        ident = rec.identity
        override = overrides.get(ident.canonical)
        # A checkpoint override is itself a scored decision about a model known to
        # exist, so it stands in for both gates below. That is how a run which
        # finished but whose row ended 'failed' gets in: is_trained sees only the
        # frozen archive trees and status == 'finished', and such a run reached
        # models/ by backfill.
        pinned = override is not None and bool(override.checkpoint)
        if not rec.is_trained and not pinned:
            continue
        # The blessing gate. Without a DB-recorded best_checkpoint there is no
        # scored decision about which kind wins, and this tree exists to publish
        # decisions -- not to mirror the store.
        if not rec.best_basename and not allow_unblessed and not pinned:
            if rec.dirs:
                gaps.append(Gap(ident.canonical, "not-db-blessed",
                                f"db={rec.db_source or 'none'} "
                                f"status={rec.db_status or '-'}"))
            continue
        if ident.legacy:
            gaps.append(Gap(ident.canonical, "legacy",
                            f"routed to models/_legacy/{ident.notes}"))
            continue
        # A rename changes only where the entry is published; the checkpoint is still
        # read from the source model's own dir (store_relpath keys on ident, not this).
        published = ident
        if override is not None and override.publish_as:
            published = replace(ident, canonical=override.publish_as)
        relpath = published.experiment_relpath
        if relpath is None:
            gaps.append(Gap(ident.canonical, "undecomposed",
                            f"arch={ident.arch} protocol={ident.protocol}"))
            continue
        hit = next((p for p in excludes if fnmatch.fnmatchcase(relpath, p)), None)
        if hit is not None:
            used_excludes.add(hit)
            gaps.append(Gap(ident.canonical, "excluded", f"{hit}: {excludes[hit]}"))
            continue
        if not rec.dirs and not (store_root / ident.store_relpath).is_dir():
            gaps.append(Gap(ident.canonical, "no-dir",
                            f"db={rec.db_source} status={rec.db_status} "
                            f"best={rec.best_basename}"))
            continue

        basename, kind, rule, note = _resolve_kind(rec, store_root, override)
        if basename is None:
            gaps.append(Gap(ident.canonical, "unresolved", rule))
            continue
        target = store_root / ident.store_relpath / basename
        if not target.exists():
            gaps.append(Gap(ident.canonical, "target-missing", str(target)))
            continue

        score = rec.best_score
        source = ""
        if override is not None:
            used.add(ident.canonical)
            if override.score is not None:
                score = override.score
            if override.publish_as:
                source = ident.canonical
                note = f"published as {override.publish_as} from {source}: {override.reason}"
        entries.append(Entry(
            canonical=published.canonical, relpath=relpath, target=target,
            kind=kind, rule=rule, db_source=rec.db_source or "",
            best_score=score, note=note, source=source))

    # A rename takes over its name: the model that already carried it steps aside, so
    # each name has one entry. Decided only after the rename itself resolved -- a rename
    # whose checkpoint is missing must not take the original down with it.
    renamed = {e.canonical: e.source for e in entries if e.source}
    kept: list[Entry] = []
    for e in entries:
        if not e.source and e.canonical in renamed:
            gaps.append(Gap(e.canonical, "superseded",
                            f"published from {renamed[e.canonical]} instead"))
            continue
        kept.append(e)
        by_relpath[e.relpath].append(e.canonical)
    entries = kept

    for relpath, owners in by_relpath.items():
        if len(owners) > 1:
            gaps.append(Gap(",".join(owners), "relpath-collision", relpath))
    # An override or exclude that matched nothing is a typo or a retired model -- say
    # so rather than let it sit in the CSV looking like it does something.
    for canonical in sorted(set(overrides) - used):
        ov = overrides[canonical]
        gaps.append(Gap(canonical, "override-unused",
                        f"{ov.checkpoint or ov.publish_as}: not published by this plan"))
    for pattern in sorted(set(excludes) - used_excludes):
        gaps.append(Gap(pattern, "exclude-unused", excludes[pattern]))
    return entries, gaps


def materialise(entries: list[Entry], staging: Path, final: Path) -> None:
    """Write the tree into ``staging`` as relative symlinks.

    Relative, not absolute, so the pair of trees can be moved together (or the
    mount renamed) without every link going stale.
    """
    for e in entries:
        link = staging / e.relpath
        link.parent.mkdir(parents=True, exist_ok=True)
        # Compute the link body relative to where the link will FINALLY live, not
        # to the staging dir -- otherwise every link breaks on promotion.
        final_link_dir = (final / e.relpath).parent
        body = os.path.relpath(e.target, final_link_dir)
        tmp = link.with_name(link.name + ".staging")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(body, tmp)
        os.replace(tmp, link)


def sync_into_place(staging: Path, final: Path, dry_run: bool) -> int:
    """rsync the staged tree over the live one, pruning stale symlinks.

    This is the **only** ``--delete`` in this package. It removes symlinks and
    empty dirs -- never a checkpoint -- so a model that is retired or re-blessed
    stops appearing here instead of lingering as a wrong answer.
    """
    final.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-rlt", "--no-perms", "--no-owner", "--no-group",
        "--info=stats2", "--delete", "--prune-empty-dirs",
        f"{staging}/", f"{final}/",
    ]
    if dry_run:
        cmd += ["--dry-run", "--itemize-changes"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.returncode


def write_manifest(entries: list[Entry], gaps: list[Gap], out_dir: Path,
                   manifest_in_tree: Optional[Path] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [["model", "relpath", "arch", "protocol", "norm", "kind", "rule",
             "db_source", "best_score", "note", "target"]]
    for e in sorted(entries, key=lambda e: e.relpath):
        parts = e.relpath.split("/")
        arch, protocol = parts[0], parts[1]
        norm = parts[2] if len(parts) == 4 else ""
        rows.append([e.canonical, e.relpath, arch, protocol, norm, e.kind, e.rule,
                     e.db_source, "" if e.best_score is None else f"{e.best_score}",
                     e.note, str(e.target)])
    for path in filter(None, (out_dir / "06_experiments_manifest.csv", manifest_in_tree)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            _csv.writer(fh).writerows(rows)

    with (out_dir / "06_experiments_gaps.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["model", "why", "detail"])
        for g in sorted(gaps, key=lambda g: (g.why, g.canonical)):
            w.writerow([g.canonical, g.why, g.detail])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--store", type=Path, default=STORE_ROOT)
    ap.add_argument("--dest", type=Path, default=EXPERIMENTS_ROOT)
    ap.add_argument("--from-roots", nargs="*", default=["data4t-slurm", "data4t-aircc"])
    ap.add_argument("--allow-unblessed", action="store_true",
                    help="also publish models with no DB best_checkpoint, resolved "
                         "from AA sweep scores")
    ap.add_argument("--overrides", type=Path, default=OVERRIDES_CSV,
                    help="checked-in per-model checkpoint/name corrections (default: %(default)s)")
    ap.add_argument("--excludes", type=Path, default=EXCLUDES_CSV,
                    help="checked-in relpath globs never to publish (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--check", action="store_true",
                    help="verify the live tree against the manifest and exit")
    # sync_into_place() runs --delete, so an under-populated plan does not just publish
    # less -- it PRUNES the live tree. The plan shrinks whenever a job DB reads as empty,
    # and census._read_db returns [] for a path that merely does not exist, which is the
    # normal appearance of ~/slurm_mount when the sshfs has dropped. Unattended callers
    # pass a floor derived from what is already on disk so that failure mode aborts
    # instead of quietly gutting the zoo. Left off by default: a hand-run prune after
    # retiring an arch is legitimate and should not need an override.
    ap.add_argument("--min-entries", type=int, default=None,
                    help="refuse to apply a plan with fewer than N entries")
    args = ap.parse_args(argv)

    if args.check:
        return _check(args.dest, args.out_dir)

    if not args.store.is_dir():
        print(f"[zoo] ERROR: {args.store} does not exist -- run Step 3 first",
              file=sys.stderr)
        return 1

    # Loaded before anything else so a malformed overrides/excludes CSV aborts the run
    # instead of letting the rebuild revert the corrected models to their DB record.
    overrides = load_overrides(args.overrides)
    excludes = load_excludes(args.excludes)
    records = build(roots=args.from_roots)
    entries, gaps = plan(records, args.store, allow_unblessed=args.allow_unblessed,
                         overrides=overrides, excludes=excludes)

    by_rule: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    for e in entries:
        by_rule[e.rule] += 1
        by_kind[e.kind] += 1
    print(f"[zoo] {_now()} {len(entries)} entries, {len(gaps)} gaps")
    print(f"[zoo]   by rule: {dict(sorted(by_rule.items()))}")
    print(f"[zoo]   by kind: {dict(sorted(by_kind.items()))}")
    gap_kinds: dict[str, int] = defaultdict(int)
    for g in gaps:
        gap_kinds[g.why] += 1
    print(f"[zoo]   gaps   : {dict(sorted(gap_kinds.items()))}")
    noted = [e for e in entries if e.note]
    if noted:
        print(f"[zoo]   {len(noted)} entr(ies) resolved by a non-db rule:")
        for e in noted[:15]:
            print(f"[zoo]     {e.canonical}: {e.kind} [{e.rule}] ({e.note})")

    if args.min_entries is not None and len(entries) < args.min_entries:
        print(f"[zoo] REFUSING: planned {len(entries)} entries, below the "
              f"--min-entries floor of {args.min_entries}. Applying this would "
              f"--delete the difference out of {args.dest}.", file=sys.stderr)
        print(f"[zoo] Usually a job DB that read as empty -- check that "
              f"{SJM_DB} is readable (is ~/slurm_mount up?).", file=sys.stderr)
        print("[zoo] If the shrink is intentional (an arch was retired), re-run by "
              "hand: model_store/scripts/ms_run.sh zoo-apply", file=sys.stderr)
        return 4

    staging = Path(tempfile.mkdtemp(prefix="ms_zoo_", dir=str(args.dest.parent)))
    try:
        materialise(entries, staging, args.dest)
        write_manifest(entries, gaps, args.out_dir,
                       manifest_in_tree=staging / "manifest.csv")
        rc = sync_into_place(staging, args.dest, dry_run=not args.apply)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if not args.apply:
        print(f"[zoo] {_now()} DRY RUN -- nothing written. Re-run with --apply.")
    else:
        dangling = [p for p in args.dest.rglob("*.pth.tar") if not p.exists()]
        print(f"[zoo] {_now()} done rc={rc}, dangling symlinks: {len(dangling)}")
        for p in dangling[:10]:
            print(f"[zoo]   DANGLING {p}", file=sys.stderr)
        if dangling:
            rc = rc or 1
    return rc


def _check(dest: Path, out_dir: Path) -> int:
    """Verify every manifest row still resolves, and nothing extra is present."""
    manifest = dest / "manifest.csv"
    if not manifest.exists():
        print(f"[zoo] ERROR: {manifest} missing", file=sys.stderr)
        return 1
    expected: set[str] = set()
    bad = 0
    with manifest.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            rel = row["relpath"]
            expected.add(rel)
            link = dest / rel
            if not link.is_symlink():
                print(f"[zoo] NOT A SYMLINK {rel}", file=sys.stderr); bad += 1
            elif not link.exists():
                print(f"[zoo] DANGLING      {rel}", file=sys.stderr); bad += 1
            elif os.path.realpath(link) != os.path.realpath(row["target"]):
                print(f"[zoo] WRONG TARGET  {rel} -> {os.path.realpath(link)}",
                      file=sys.stderr); bad += 1
    actual = {str(p.relative_to(dest)) for p in dest.rglob("*.pth.tar")}
    for extra in sorted(actual - expected):
        print(f"[zoo] UNMANAGED     {extra}", file=sys.stderr); bad += 1
    print(f"[zoo] checked {len(expected)} manifest rows, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
