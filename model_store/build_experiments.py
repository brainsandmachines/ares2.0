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
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .census import EXPERIMENTS_ROOT, SJM_DB, STORE_ROOT, ModelRecord, build
from .dedupe_report import LOG_DIR, _now
from .naming import CKPT_FILE_FOR_KIND, KIND_FOR_CKPT_FILE

GIB = 1024 ** 3


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


@dataclass
class Gap:
    canonical: str
    why: str
    detail: str = ""


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
    rec: ModelRecord, store_root: Path,
) -> tuple[Optional[str], str, str, str]:
    """Return ``(checkpoint basename, kind, rule, note)`` or ``(None, "", reason, "")``."""
    model_dir = store_root / rec.identity.store_relpath

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
) -> tuple[list[Entry], list[Gap]]:
    entries: list[Entry] = []
    gaps: list[Gap] = []
    by_relpath: dict[str, list[str]] = defaultdict(list)

    for rec in sorted(records.values(), key=lambda r: r.identity.canonical):
        ident = rec.identity
        if not rec.is_trained:
            continue
        # The blessing gate. Without a DB-recorded best_checkpoint there is no
        # scored decision about which kind wins, and this tree exists to publish
        # decisions -- not to mirror the store.
        if not rec.best_basename and not allow_unblessed:
            if rec.dirs:
                gaps.append(Gap(ident.canonical, "not-db-blessed",
                                f"db={rec.db_source or 'none'} "
                                f"status={rec.db_status or '-'}"))
            continue
        if ident.legacy:
            gaps.append(Gap(ident.canonical, "legacy",
                            f"routed to models/_legacy/{ident.notes}"))
            continue
        relpath = ident.experiment_relpath
        if relpath is None:
            gaps.append(Gap(ident.canonical, "undecomposed",
                            f"arch={ident.arch} protocol={ident.protocol}"))
            continue
        if not rec.dirs and not (store_root / ident.store_relpath).is_dir():
            gaps.append(Gap(ident.canonical, "no-dir",
                            f"db={rec.db_source} status={rec.db_status} "
                            f"best={rec.best_basename}"))
            continue

        basename, kind, rule, note = _resolve_kind(rec, store_root)
        if basename is None:
            gaps.append(Gap(ident.canonical, "unresolved", rule))
            continue
        target = store_root / ident.store_relpath / basename
        if not target.exists():
            gaps.append(Gap(ident.canonical, "target-missing", str(target)))
            continue

        entries.append(Entry(
            canonical=ident.canonical, relpath=relpath, target=target,
            kind=kind, rule=rule, db_source=rec.db_source or "",
            best_score=rec.best_score, note=note))
        by_relpath[relpath].append(ident.canonical)

    for relpath, owners in by_relpath.items():
        if len(owners) > 1:
            gaps.append(Gap(",".join(owners), "relpath-collision", relpath))
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

    records = build(roots=args.from_roots)
    entries, gaps = plan(records, args.store,
                         allow_unblessed=args.allow_unblessed)

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
        print(f"[zoo]   {len(noted)} entr(ies) resolved without a DB score:")
        for e in noted[:15]:
            print(f"[zoo]     {e.canonical}: {e.kind} ({e.note})")

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
