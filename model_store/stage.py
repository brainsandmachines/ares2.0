"""Steps 7 and 8: stage the leftovers for deletion -- by ``mv``, never ``rm``.

Nothing here deletes. Files are **renamed** into ``pending_deletion/<date>/``,
preserving their relative path, with a ``MANIFEST.csv`` and a ``README.md``
explaining what they are. A same-filesystem rename moves no data, takes no time,
and is fully reversible; the user erases the staging dir by hand.

That also means staging frees **zero** bytes on its own -- the space comes back
only when the user removes the staging dir. This is deliberate: the alternative
(``rsync --delete``) is an irreversible erase, which is the opposite of "everything
should be pending deletion, I erase it myself".

**The data4t pass is self-verifying.** Step 3 hardlinked every keeper into
``models/``, so after it every keeper has ``nlink >= 2``. A file under the archives
with ``nlink == 1`` is therefore provably *not* referenced from ``models/`` -- the
filesystem decides what is safe to stage, not a list this program has to be
trusted to compute. Before moving anything the pass also refuses if it finds an
unlinked keeper checkpoint it cannot account for (see ``_unexplained_keepers``),
because that would mean Step 3 missed a model.

The /mnt/data pass stages only files that ``model_store.promotions`` matched to an
archive file **by sha256**. Name equality is never accepted as proof.
"""

from __future__ import annotations

import argparse
import csv as _csv
import datetime as _dt
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .census import ARCHIVE_ROOTS, DATA4T_ROOT, STORE_ROOT, build
from .dedupe_report import LOG_DIR, _now
from .naming import PERIODIC_RE, is_intermediate, is_keeper_checkpoint
from .promotions import DATA_ROOT, PROMOTED_ROOT, THIRD_PARTY_ROOT

GIB = 1024 ** 3
TODAY = _dt.date.today().isoformat()

# The two local archive roots, as *archive* dirs (one level above models/).
LOCAL_ARCHIVES = {
    "aircc_archive": DATA4T_ROOT / "aircc_archive",
    "slurm_archive": DATA4T_ROOT / "slurm_archive",
}

README = """\
# pending_deletion/{date}

Staged by `model_store.stage` on {now}. **Nothing here has been deleted.**
Every file was moved (renamed) out of its original tree, which is why this
directory currently frees no space -- the space comes back when *you* remove it.

## What is in here

{sections}

## Why it is safe to erase

Every file in this directory has been checked to be unreferenced from the curated
tree:

* files staged from the archives had a hardlink count of 1 at the time of the
  move, meaning `/mnt/data4t/models` does **not** point at them (every keeper it
  does point at has a count of 2 or more);
* files staged from `/mnt/data` were matched to an archive file by **sha256**, so
  a byte-identical copy demonstrably remains.

You can re-verify either claim yourself at any time:

```bash
# nothing in here is reachable from the curated tree
python -m model_store.stage --check-disjoint {root}

# spot-check one staged checkpoint against its surviving twin
sha256sum {root}/<some>/<file>.pth.tar
grep -F "<file>" {root}/MANIFEST.csv     # names the surviving copy
```

## To erase

```bash
rm -rf {root}
```

## To put it back

```bash
python -m model_store.stage --unstage {root}
```
"""


@dataclass
class StageItem:
    src: Path
    rel: str            # path relative to the staging root
    size: int
    reason: str
    survives_at: str = ""


# --------------------------------------------------------------------------
# data4t: the archives
# --------------------------------------------------------------------------
def _iter_archive_files(archive: Path) -> Iterable[tuple[Path, os.stat_result]]:
    for dirpath, _dirnames, filenames in os.walk(archive):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            yield path, st


def _reason_for(path: Path, archive: Path) -> str:
    rel = path.relative_to(archive)
    top = rel.parts[0] if rel.parts else ""
    if top == "models_failed":
        return "models_failed"
    if is_intermediate(path.name):
        return "intermediate-checkpoint"
    if is_keeper_checkpoint(path.name):
        return "superseded-keeper"
    return "unreferenced-metadata"


def _is_model_artifact(rel: Path) -> bool:
    """True for a file that belongs to one model run, and so is ours to stage.

    ``models/<model>/<file>`` (any depth below the model) and everything under
    ``models_failed/`` qualify. Anything shallower -- a file sitting at the
    archive root or directly in ``models/`` -- is archive-level provenance that
    no model dir owns; see the note in :func:`plan_data4t`.
    """
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == "models_failed":
        return True
    return parts[0] == "models" and len(parts) >= 3


def plan_data4t(
    store_root: Path, archives: dict[str, Path], min_store_files: int,
) -> tuple[list[StageItem], list[str]]:
    """Everything under the archives with ``nlink == 1``, plus refusal reasons."""
    problems: list[str] = []

    if not store_root.is_dir():
        problems.append(f"{store_root} does not exist -- run Step 3 (build_models) first")
        return [], problems
    store_files = sum(1 for _ in store_root.rglob("*") if _.is_file())
    if store_files < min_store_files:
        problems.append(
            f"{store_root} holds only {store_files} files (expected >= {min_store_files}). "
            f"Refusing: if Step 3 did not finish, every archive file looks unreferenced "
            f"and this pass would stage the entire archive.")
        return [], problems

    items: list[StageItem] = []
    for label, archive in sorted(archives.items()):
        if not archive.is_dir():
            continue
        for path, st in _iter_archive_files(archive):
            if st.st_nlink > 1:
                continue        # referenced from models/ -- keep
            rel = path.relative_to(archive)
            if not _is_model_artifact(rel):
                # Archive-level provenance, not a model artifact: the job-DB
                # snapshots, the prune manifests, the runtime-comparison runs,
                # the old helper scripts. The nlink test says nothing useful
                # about these -- `models/` was never going to link them -- and
                # the AIRCC snapshot in particular is what blesses every AIRCC
                # entry in the zoo, so staging it for deletion would quietly
                # degrade the next rebuild to SJM rows only. They belong in
                # ``models/_meta/`` instead; see census.AIRCC_DB.
                continue
            items.append(StageItem(
                src=path, rel=f"{label}/{rel}", size=st.st_size,
                reason=_reason_for(path, archive)))
    return items, problems


def _explained_unlinked(
    decisions_csv: Path, build_csv: Path, backfill_csv: Path,
) -> set[str]:
    """Archive paths that are legitimately unreferenced from ``models/``.

    Three ways a keeper checkpoint ends up at ``nlink == 1`` on purpose:

    * it is the **redundant half of a byte-identical pair** -- 248 checkpoints
      exist on both the local Slurm and the local AIRCC side with the same
      sha256, and Step 3 hardlinked exactly one of them. The other was never
      linked, and staging it loses nothing because its twin *is* the curated
      file (this is the common case by a wide margin);
    * it **lost a merge conflict** -- Step 3 hardlinked the other side, so the
      loser named in the approved decisions was never linked;
    * it **was superseded by the QNAP backfill** -- Step 4 replaced the curated
      copy with a higher-epoch one, and rsync's write-temp-then-rename broke the
      hardlink, leaving the original archive file with a single link.

    None of the three is taken on the CSV's word alone: a path is only explained
    once the curated file that replaces it is confirmed to exist on disk. So a
    model Step 3 skipped outright still has no curated destination, stays
    unexplained, and makes the pass refuse -- which is the whole point of the
    check.
    """
    explained: set[str] = set()

    # dest <-> source, both directions, from the Step 3 plan.
    source_of: dict[str, str] = {}
    dest_of: dict[str, str] = {}
    if build_csv.exists():
        with build_csv.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                source_of[row["dest_relpath"]] = row["source_path"]
                dest_of[row["source_path"]] = row["dest_relpath"]

    def _confirm(side: str, twin: str) -> None:
        """Explain ``side`` only if the curated file that stands in for it exists.

        The curated file is normally the one Step 3 linked from ``twin``; when
        ``side`` itself was the linked source and the backfill later overwrote
        it, the destination is registered under ``side`` instead. Either way the
        test is the same -- a real file at the curated path.
        """
        dest = dest_of.get(twin) or dest_of.get(side)
        if dest and Path(dest).is_file():
            explained.add(side)

    if decisions_csv.exists():
        with decisions_csv.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                left = (row.get("left_path") or "").strip()
                right = (row.get("right_path") or "").strip()
                if row.get("verdict") == "IDENTICAL" and left and right:
                    # sha256-proven equal, so whichever side went unlinked is a
                    # pure duplicate of what the curated tree already holds.
                    _confirm(left, right)
                    _confirm(right, left)
                    continue
                loser = (row.get("loser_path") or "").strip()
                if loser:
                    _confirm(loser, right if loser == left else left)

    # Files with no decision row at all (present under a single root) whose
    # curated copy the backfill replaced with a higher-epoch QNAP one.
    if backfill_csv.exists():
        with backfill_csv.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                if row.get("reason") == "missing":
                    continue
                src = source_of.get(row["dest"])
                if src and Path(row["dest"]).is_file():
                    explained.add(src)
    return explained


def _unexplained_keepers(
    items: list[StageItem], decisions_csv: Path, build_csv: Path, backfill_csv: Path,
) -> list[StageItem]:
    """Unlinked keeper checkpoints nothing accounts for -- a reason to refuse."""
    approved_paths = _explained_unlinked(decisions_csv, build_csv, backfill_csv)

    out: list[StageItem] = []
    for item in items:
        if item.reason != "superseded-keeper":
            continue
        if "/models_failed/" in item.rel:
            continue
        if str(item.src) in approved_paths:
            continue
        out.append(item)
    return out


# --------------------------------------------------------------------------
# /mnt/data: the promoted zoo
# --------------------------------------------------------------------------
# Paths under /mnt/data that something still reads, and so must not be staged even
# when a byte-identical copy exists in the archive. `epsilon_bounded_contstim`'s
# conf/machine/botero.yaml points `models_path` at the first of these, and
# conf/explicit_pairs/superclass_loss_non_gt_l2_init1_pairs.yaml names the files in
# it by their flat name -- which the curated tree does not reproduce (there they are
# `<model>/model_best.pth.tar`). 4.5 GiB, against ~150 GiB reclaimed elsewhere.
KEEP_PREFIXES = (
    DATA_ROOT / "robustness_models" / "madry" / "l2" / "init1",
)


def plan_data_drive(promotions_csv: Path) -> tuple[list[StageItem], list[str]]:
    problems: list[str] = []
    if not promotions_csv.exists():
        problems.append(f"{promotions_csv} missing -- run Step 5 (promotions) first")
        return [], problems

    items: list[StageItem] = []
    with promotions_csv.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("status") != "MATCHED":
                continue
            src = Path(row["promoted_path"])
            if not src.exists() or src.is_symlink():
                continue
            if any(str(src).startswith(str(k) + "/") for k in KEEP_PREFIXES):
                continue
            try:
                rel = src.relative_to(DATA_ROOT)
            except ValueError:
                continue
            items.append(StageItem(
                src=src, rel=str(rel), size=int(row["size"] or 0),
                reason="promoted-duplicate", survives_at=row.get("matched_path", "")))
    return items, problems


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------
def apply_stage(items: list[StageItem], root: Path, dry_run: bool) -> int:
    if dry_run:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for item in items:
        dest = root / item.rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        try:
            os.rename(item.src, dest)       # same filesystem: instant, no copy
            moved += 1
        except OSError as exc:
            print(f"[stage] ERROR moving {item.src}: {exc}", file=sys.stderr)
            return 1
    print(f"[stage] {_now()} moved {moved} files into {root}")
    return 0


def write_manifest(items: list[StageItem], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "MANIFEST.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["staged_relpath", "original_path", "size", "reason", "survives_at"])
        for item in sorted(items, key=lambda i: i.rel):
            w.writerow([item.rel, item.src, item.size, item.reason, item.survives_at])

    by_reason: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for item in items:
        n, b = by_reason[item.reason]
        by_reason[item.reason] = (n + 1, b + item.size)
    sections = "\n".join(
        f"* **{reason}** -- {n} files, {b / GIB:.1f} GiB"
        for reason, (n, b) in sorted(by_reason.items(), key=lambda kv: -kv[1][1])
    )
    (root / "README.md").write_text(README.format(
        date=root.name, now=_now(), sections=sections, root=root))


def check_disjoint(root: Path, store_root: Path) -> int:
    """Prove nothing under ``root`` is reachable from the curated tree."""
    if not root.is_dir():
        print(f"[stage] {root} does not exist", file=sys.stderr)
        return 1
    store_inodes: set[tuple[int, int]] = set()
    for path in store_root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                st = path.stat()
                store_inodes.add((st.st_dev, st.st_ino))
        except OSError:
            continue
    shared = []
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                st = path.stat()
                if (st.st_dev, st.st_ino) in store_inodes:
                    shared.append(path)
        except OSError:
            continue
    print(f"[stage] {store_root}: {len(store_inodes)} distinct inodes")
    print(f"[stage] {root}: {len(shared)} file(s) also reachable from the curated tree")
    for path in shared[:20]:
        print(f"[stage]   SHARED {path}", file=sys.stderr)
    return 1 if shared else 0


def unstage(root: Path, dry_run: bool) -> int:
    """Move a staged tree back where it came from, using its MANIFEST.csv."""
    manifest = root / "MANIFEST.csv"
    if not manifest.exists():
        print(f"[stage] ERROR: {manifest} missing", file=sys.stderr)
        return 1
    restored = 0
    with manifest.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            src = root / row["staged_relpath"]
            dest = Path(row["original_path"])
            if not src.exists() or dest.exists():
                continue
            if dry_run:
                print(f"[stage] DRY restore {src} -> {dest}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dest)
            restored += 1
    print(f"[stage] {_now()} restored {restored} files from {root}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scope", choices=("data4t", "data"),
                    help="which tree to stage from")
    ap.add_argument("--apply", action="store_true", help="move (default: dry run)")
    ap.add_argument("--store", type=Path, default=STORE_ROOT)
    ap.add_argument("--root", type=Path, help="staging root (default: <disk>/pending_deletion/<date>)")
    ap.add_argument("--decisions", type=Path, default=LOG_DIR / "02_merge_decisions.csv")
    ap.add_argument("--build-plan", type=Path, default=LOG_DIR / "03_build_plan.csv")
    ap.add_argument("--backfill-plan", type=Path, default=LOG_DIR / "04_backfill_plan.csv")
    ap.add_argument("--promotions", type=Path, default=LOG_DIR / "05_promotions.csv")
    ap.add_argument("--min-store-files", type=int, default=2000,
                    help="refuse the data4t pass if models/ holds fewer files than this")
    ap.add_argument("--check-disjoint", type=Path, metavar="ROOT",
                    help="verify a staging root shares no inode with models/")
    ap.add_argument("--unstage", type=Path, metavar="ROOT", help="undo a staging pass")
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR)
    args = ap.parse_args(argv)

    if args.check_disjoint:
        return check_disjoint(args.check_disjoint, args.store)
    if args.unstage:
        return unstage(args.unstage, dry_run=not args.apply)
    if not args.scope:
        ap.error("one of --scope, --check-disjoint or --unstage is required")

    if args.scope == "data4t":
        root = args.root or DATA4T_ROOT / "pending_deletion" / TODAY
        items, problems = plan_data4t(args.store, LOCAL_ARCHIVES, args.min_store_files)
        if not problems:
            unexplained = _unexplained_keepers(
                items, args.decisions, args.build_plan, args.backfill_plan)
            if unexplained:
                problems.append(
                    f"{len(unexplained)} unlinked keeper checkpoint(s) are not accounted "
                    f"for by the approved decisions -- Step 3 may have skipped a model")
                for item in unexplained[:15]:
                    problems.append(f"    {item.src}")
    else:
        root = args.root or DATA_ROOT / "pending_deletion" / TODAY
        items, problems = plan_data_drive(args.promotions)

    if problems:
        print(f"[stage] REFUSING ({len(problems)} problem(s)):", file=sys.stderr)
        for p in problems:
            print(f"[stage]   {p}", file=sys.stderr)
        return 3

    by_reason: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for item in items:
        n, b = by_reason[item.reason]
        by_reason[item.reason] = (n + 1, b + item.size)
    total = sum(i.size for i in items)
    print(f"[stage] {_now()} scope={args.scope} -> {root}")
    for reason, (n, b) in sorted(by_reason.items(), key=lambda kv: -kv[1][1]):
        print(f"[stage]   {n:6d} files  {b / GIB:9.1f} GiB  {reason}")
    print(f"[stage]   {len(items):6d} files  {total / GIB:9.1f} GiB  TOTAL "
          f"(reclaimed only when you erase {root})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "07" if args.scope == "data4t" else "08"
    plan_csv = args.out_dir / f"{tag}_stage_{args.scope}.csv"
    with plan_csv.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["staged_relpath", "original_path", "size", "reason", "survives_at"])
        for item in sorted(items, key=lambda i: i.rel):
            w.writerow([item.rel, item.src, item.size, item.reason, item.survives_at])
    print(f"[stage] wrote {plan_csv}")

    if not args.apply:
        print(f"[stage] {_now()} DRY RUN -- nothing moved. Re-run with --apply.")
        return 0

    rc = apply_stage(items, root, dry_run=False)
    if rc == 0:
        write_manifest(items, root)
        if args.scope == "data4t":
            rc = check_disjoint(root, args.store)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
