"""Step 3: build ``/mnt/data4t/models`` -- the curated copy -- by hardlink.

One model dir per model, named so the arch is always part of the name, holding
every keeper file and **no** ``checkpoint-N``:

    models/convnext_base/convnext_base_l2_4_init1/{last,model_best,model_best_adv}.pth.tar
    models/swin_b/swin_b_l2_4_init1/...          (was swin_b/l2_4_init1)
    models/_legacy/old_models/convnext_small_l2_1_init1/...

**Hardlinks, not copies.** Source and destination are both on ``/dev/sda1``, so the
whole tree costs directory entries and nothing else -- no 1.2 TB of duplication and
no re-pull over CIFS. Verified: ``rsync -rt --link-dest=<srcdir> --files-from=<list>
<srcdir>/ <destdir>/`` produces same-inode files with ``nlink=2``, skips the
intermediates the list omits, and is a no-op on re-run.

That choice also makes Step 7 self-verifying: after this pass every keeper has
``nlink >= 2``, so ``find <archive> -type f -links 1`` is exactly the set of files
*not* referenced from ``models/`` -- i.e. the discards, identified by the
filesystem rather than by a list this program has to be trusted to get right.

Conflicts (a model present in both archives) are resolved from the **approved**
decision CSV that Step 2 produced. Without it this pass refuses to touch any
multi-root model, because guessing there is exactly what the gate exists to
prevent.

Cross-device sources (the QNAP) cannot be hardlinked and are reported for Step 4's
backfill instead of being silently copied here.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .census import ARCHIVE_ROOTS, STORE_ROOT, ModelRecord, build
from .dedupe_report import LOG_DIR, _now
from .naming import is_intermediate

GIB = 1024 ** 3

# Preference order when a file is present under several roots and no decision
# applies: local before QNAP (hardlinkable), slurm before aircc (the slurm archive
# is already curated, so it is the cleaner base).
ROOT_PREFERENCE = ("data4t-slurm", "data4t-aircc", "qnap-slurm", "qnap-aircc")

# House rsync flags, matching backup_slurm_models.sh:211-217.
RSYNC_BASE = [
    "rsync", "-rt", "--no-perms", "--no-owner", "--no-group", "--partial",
]


@dataclass
class FilePlan:
    """One file's resolution: which root it comes from, and why."""

    relpath: str
    source_label: str
    source_path: Path
    size: int
    reason: str            # only-root | approved-winner | identical | newer-mtime
    alt_of: Optional[str] = None   # set for the _alt/<label>/ copy of a loser


@dataclass
class ModelPlan:
    canonical: str
    dest: Path
    files: list[FilePlan] = field(default_factory=list)
    skipped_cross_device: list[str] = field(default_factory=list)
    needs_decision: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------
def load_decisions(csv_path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """``(canonical, relpath)`` -> ``(verdict, winner_label)`` from Step 2's CSV."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            key = (row["model"], row["file"])
            out[key] = (row["verdict"], row.get("winner") or "")
    return out


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
def _walk_model_files(model_dir: Path) -> dict[str, tuple[Path, os.stat_result]]:
    """Every file under a model dir that the curated tree keeps, by relative path.

    Drops ``checkpoint-N.pth.tar`` and ``tmp.pth.tar``; keeps everything else,
    including ``periodic/``, ``pgd_eval*/``, ``.hydra/``, logs, configs, the
    AutoAttack sweep CSVs/JSON and the comparison plots.
    """
    out: dict[str, tuple[Path, os.stat_result]] = {}
    for dirpath, dirnames, filenames in os.walk(model_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".backup")]
        for name in filenames:
            if is_intermediate(name):
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            out[str(path.relative_to(model_dir))] = (path, st)
    return out


def plan_model(
    rec: ModelRecord, labels: list[str], decisions: dict, dest_root: Path,
    dest_dev: int, keep_alt: bool, allow_unapproved: bool = False,
) -> ModelPlan:
    canonical = rec.identity.canonical
    plan = ModelPlan(canonical=canonical, dest=dest_root / rec.identity.store_relpath)

    present = [lbl for lbl in labels if lbl in rec.dirs]
    if not present:
        return plan
    ordered = [lbl for lbl in ROOT_PREFERENCE if lbl in present]

    per_root: dict[str, dict[str, tuple[Path, os.stat_result]]] = {
        lbl: _walk_model_files(rec.dirs[lbl]) for lbl in ordered
    }
    all_rel = sorted({rel for files in per_root.values() for rel in files})

    for rel in all_rel:
        holders = [lbl for lbl in ordered if rel in per_root[lbl]]
        chosen: Optional[str] = None
        reason = ""

        if len(holders) == 1:
            chosen, reason = holders[0], "only-root"
        else:
            verdict, winner = decisions.get((canonical, rel), ("", ""))
            if verdict == "DIVERGENT" and winner in holders:
                chosen, reason = winner, "approved-winner"
            elif verdict == "IDENTICAL":
                chosen, reason = holders[0], "identical"
            elif rel.endswith(".pth.tar") and not allow_unapproved:
                # A checkpoint present in two roots with no approved decision:
                # this is precisely what the Step 2 gate covers, so refuse. Note
                # this leaves the model with *no* checkpoint in the plan, which is
                # why main() treats any needs_decision as a hard stop rather than
                # letting a metadata-only model dir get built.
                plan.needs_decision.append(rel)
                continue
            else:
                # Metadata (logs, configs, AA csv/json, plots), or a checkpoint
                # under --allow-unapproved. Newest wins, and the loser is preserved
                # under _alt/ so nothing is lost either way.
                chosen = max(holders, key=lambda lbl: per_root[lbl][rel][1].st_mtime)
                reason = "newer-mtime"

        src_path, st = per_root[chosen][rel]
        if st.st_dev != dest_dev:
            plan.skipped_cross_device.append(rel)
            continue
        plan.files.append(FilePlan(rel, chosen, src_path, st.st_size, reason))

        # Preserve the losing side only for *metadata*. A superseded log.txt or
        # summary.csv is small (~1.5 GiB across the whole tree) and irreplaceable
        # context for how a run went, so it is kept under _alt/<label>/. A
        # superseded *checkpoint* is ~1.3 GiB, is still on the QNAP, and is what
        # Step 7 is meant to reclaim -- keeping it here would negate that.
        if keep_alt and not rel.endswith(".pth.tar") and reason == "newer-mtime":
            for other in holders:
                if other == chosen:
                    continue
                o_path, o_st = per_root[other][rel]
                if o_st.st_size == st.st_size and abs(o_st.st_mtime - st.st_mtime) < 2:
                    continue    # same file, nothing to preserve
                if o_st.st_dev != dest_dev:
                    continue
                plan.files.append(FilePlan(
                    f"_alt/{other}/{rel}", other, o_path, o_st.st_size,
                    "loser-preserved", alt_of=rel))
    return plan


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------
def apply_model(plan: ModelPlan, dry_run: bool, log=sys.stdout) -> int:
    """Hardlink one model's files. Returns an rsync-style rc (0 = ok)."""
    if not plan.files:
        return 0
    # One rsync leg per (source model dir, dest prefix): --link-dest is per-source,
    # and a leg's --files-from paths are relative to that one source dir.
    legs: dict[tuple[Path, str], list[str]] = defaultdict(list)
    for fp in plan.files:
        src_dir, dest_prefix, inner = _leg_of(fp)
        legs[(src_dir, dest_prefix)].append(inner)

    rc = 0
    for (src_dir, prefix), rels in sorted(legs.items(), key=lambda kv: str(kv[0])):
        dest_dir = plan.dest / prefix if prefix else plan.dest
        if dry_run:
            print(f"[build] DRY {src_dir} -> {dest_dir}  ({len(rels)} files)", file=log)
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".files", delete=False) as fh:
            fh.write("\n".join(rels) + "\n")
            list_path = fh.name
        try:
            cmd = RSYNC_BASE + [
                f"--link-dest={src_dir}", f"--files-from={list_path}",
                f"{src_dir}/", f"{dest_dir}/",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[build] ERROR rsync rc={proc.returncode} for {plan.canonical}: "
                      f"{proc.stderr.strip()}", file=sys.stderr)
                rc = proc.returncode
        finally:
            os.unlink(list_path)
    return rc


def _leg_of(fp: FilePlan) -> tuple[Path, str, str]:
    """Split a plan entry into ``(source model dir, dest prefix, path within both)``.

    A normal entry ``summary.csv`` came from ``<srcdir>/summary.csv`` and lands at
    ``<dest>/summary.csv``: prefix empty, inner == relpath.

    A preserved-loser entry ``_alt/<label>/periodic/epoch_0090.pth.tar`` came from
    ``<srcdir>/periodic/epoch_0090.pth.tar`` and lands under ``<dest>/_alt/<label>/``:
    the prefix is the two synthetic leading components, and only the remainder is
    shared with the source path.
    """
    if fp.relpath.startswith("_alt/"):
        parts = fp.relpath.split("/", 2)
        prefix, inner = f"{parts[0]}/{parts[1]}", parts[2]
    else:
        prefix, inner = "", fp.relpath
    src_dir = fp.source_path
    for _ in range(inner.count("/") + 1):
        src_dir = src_dir.parent
    return src_dir, prefix, inner


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--dest", type=Path, default=STORE_ROOT)
    ap.add_argument("--from-roots", nargs="*", default=["data4t-slurm", "data4t-aircc"],
                    choices=sorted(ARCHIVE_ROOTS))
    ap.add_argument("--decisions", type=Path, default=LOG_DIR / "02_merge_decisions.csv")
    ap.add_argument("--allow-unapproved", action="store_true",
                    help="resolve checkpoint conflicts by mtime instead of refusing "
                         "(only for testing -- the Step 2 gate exists for a reason)")
    ap.add_argument("--no-alt", action="store_true",
                    help="do not preserve the losing side's differing metadata")
    ap.add_argument("--only", nargs="*", help="limit to these canonical model names")
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR)
    args = ap.parse_args(argv)

    dest_root = args.dest
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_dev = dest_root.stat().st_dev

    decisions = load_decisions(args.decisions)
    if not decisions and not args.allow_unapproved:
        print(f"[build] no approved decision list at {args.decisions}.", file=sys.stderr)
        print("[build] Run Step 2 (ms_run.sh conflicts) and approve its report first,",
              file=sys.stderr)
        print("[build] or pass --allow-unapproved for a dry run.", file=sys.stderr)
        return 2
    print(f"[build] {_now()} {len(decisions)} decisions loaded from {args.decisions}")

    records = build(roots=args.from_roots)
    trained = {k: r for k, r in records.items() if r.is_trained and r.dirs}
    if args.only:
        wanted = set(args.only)
        trained = {k: r for k, r in trained.items() if r.identity.canonical in wanted}
    print(f"[build] {_now()} planning {len(trained)} models -> {dest_root}")

    plans: list[ModelPlan] = []
    for rec in sorted(trained.values(), key=lambda r: r.identity.canonical):
        plans.append(plan_model(
            rec, args.from_roots, decisions, dest_root, dest_dev,
            keep_alt=not args.no_alt, allow_unapproved=args.allow_unapproved))

    blocked = [p for p in plans if p.needs_decision]
    if blocked and not args.allow_unapproved:
        print(f"[build] REFUSING: {len(blocked)} models have checkpoint conflicts with "
              f"no approved decision:", file=sys.stderr)
        for p in blocked[:20]:
            print(f"  {p.canonical}: {p.needs_decision}", file=sys.stderr)
        if len(blocked) > 20:
            print(f"  ... and {len(blocked) - 20} more", file=sys.stderr)
        return 3

    total_files = sum(len(p.files) for p in plans)
    total_bytes = sum(p.total_bytes for p in plans)
    cross = sum(len(p.skipped_cross_device) for p in plans)
    alt = sum(1 for p in plans for f in p.files if f.alt_of)
    print(f"[build] {_now()} plan: {len(plans)} models, {total_files} files, "
          f"{total_bytes / GIB:.1f} GiB of content (0 bytes copied -- hardlinks)")
    print(f"[build] {_now()}   {alt} loser-metadata files preserved under _alt/")
    if cross:
        print(f"[build] {_now()}   {cross} files skipped as cross-device "
              f"(need Step 4 backfill)")

    _write_plan_csv(plans, args.out_dir / "03_build_plan.csv")
    print(f"[build] wrote {args.out_dir / '03_build_plan.csv'}")

    if not args.apply:
        print(f"[build] {_now()} DRY RUN -- nothing written. Re-run with --apply.")
        return 0

    rc = 0
    for idx, plan in enumerate(sorted(plans, key=lambda p: p.canonical), 1):
        leg_rc = apply_model(plan, dry_run=False)
        rc = leg_rc or rc
        if idx % 25 == 0 or idx == len(plans):
            print(f"[build] {_now()} {idx}/{len(plans)} models linked", flush=True)
    print(f"[build] {_now()} done rc={rc}")
    return rc


def _write_plan_csv(plans: list[ModelPlan], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["model", "dest_relpath", "source_label", "source_path",
                    "size", "reason", "alt_of"])
        for p in plans:
            for f in p.files:
                w.writerow([p.canonical, f"{p.dest}/{f.relpath}", f.source_label,
                            f.source_path, f.size, f.reason, f.alt_of or ""])


if __name__ == "__main__":
    raise SystemExit(main())
