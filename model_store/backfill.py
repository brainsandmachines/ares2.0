"""Step 4: pull from the QNAP whatever the curated tree is missing.

**This is route 2**, the second of the two weekly rsync routes:

    slurm results/models --(Sun, backup_slurm_models.sh)--> /mnt/botero/slurm_archive
    /mnt/botero/slurm_archive --(Mon, this pass)--> /mnt/data4t/models
                                                    --> models_for_experiments symlinks

Since the weekly backup was repointed to write straight to the QNAP, the local
``/mnt/data4t/{slurm,aircc}_archive`` trees Step 3 hardlinks from are frozen, and
the QNAP is the only live source the curated tree has. So ``models/`` built from
local sources alone is now permanently incomplete, and this pass is what closes the
gap -- weekly, from ``model_store/scripts/ms_weekly_sync.sh``, with
``--roots qnap-slurm``.

Unlike Step 3 these are **real copies** -- the QNAP is a different
filesystem, so hardlinking is impossible and the bytes actually land on
``/dev/sda1``. That makes this the one pass with a space cost, which is why it
reports its exact size and refuses to start without headroom.

Selection never overrides an approved merge decision. A file is pulled when it is
absent locally; when it is a **checkpoint** whose QNAP copy records a *higher
epoch*; or when it is **metadata** whose QNAP copy is newer. Size and mtime alone
are not enough for a checkpoint -- several QNAP-AIRCC copies are newer but far less
trained (epoch 6 against 199), so a size/mtime rule would quietly undo the epoch
decisions Step 2 gated on. ``-rt`` is preserved throughout so that mtime stays a
usable signal for every other consumer of these trees.
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

from .census import ARCHIVE_ROOTS, QNAP_ROOT, STORE_ROOT, build
from .dedupe_report import LOG_DIR, _now
from .epochs import checkpoint_epoch
from .naming import is_intermediate

GIB = 1024 ** 3
QNAP_LABELS = ("qnap-slurm", "qnap-aircc")

# Refuse to start a pull that would leave less than this free.
MIN_FREE_GB = int(os.environ.get("MS_BACKFILL_MIN_FREE_GB", "150"))

# NOTE: no --inplace and no --append, deliberately. Step 3 made every file under
# models/ a hardlink to the archive copy, so writing *through* a destination file
# would modify the archive's bytes as well -- silently corrupting the very copy we
# are treating as the master. rsync's default (write a temp file, then rename over
# the destination) breaks the link instead, leaving the archive untouched and its
# now-superseded copy at nlink 1, which is exactly how Step 7 recognises a discard.
RSYNC_BASE = [
    "rsync", "-rt", "--no-perms", "--no-owner", "--no-group", "--partial",
    "--info=stats2",
]

# Only these subtrees are curated; models_failed and the QNAP-only staged dir are
# explicitly out of scope (the user chose to drop models_failed from data4t).
SKIP_TOP = ("models_failed", "_pending_delete_20260810")


@dataclass
class PullItem:
    canonical: str
    source_label: str
    source: Path
    dest: Path
    rel: str
    size: int
    reason: str        # missing | qnap-higher-epoch | qnap-newer
    local_epoch: Optional[int] = None
    qnap_epoch: Optional[int] = None


def plan(records: dict, store_root: Path,
         labels: tuple[str, ...] = QNAP_LABELS) -> list[PullItem]:
    items: list[PullItem] = []
    for rec in sorted(records.values(), key=lambda r: r.identity.canonical):
        dest_dir = store_root / rec.identity.store_relpath
        for label in labels:
            src_dir = rec.dirs.get(label)
            if src_dir is None:
                continue
            for dirpath, dirnames, filenames in os.walk(src_dir):
                dirnames[:] = [d for d in dirnames if not d.startswith(".backup")]
                for name in filenames:
                    if is_intermediate(name):
                        continue
                    src = Path(dirpath) / name
                    if src.is_symlink():
                        continue
                    try:
                        sst = src.stat()
                    except OSError:
                        continue
                    rel = str(src.relative_to(src_dir))
                    dest = dest_dir / rel
                    try:
                        dst = dest.stat()
                    except OSError:
                        items.append(PullItem(
                            rec.identity.canonical, label, src, dest, rel,
                            sst.st_size, "missing"))
                        continue
                    if dst.st_size == sst.st_size and abs(sst.st_mtime - dst.st_mtime) < 2:
                        continue        # same file, nothing to do

                    if name.endswith(".pth.tar"):
                        # A checkpoint already in the curated tree got there by an
                        # approved decision, and that decision was made on EPOCH,
                        # not file age. Several QNAP-AIRCC copies are newer but far
                        # less trained (epoch 6 against 199), so a size/mtime rule
                        # here would silently undo the merge decisions. Only a
                        # genuinely higher epoch replaces what is already there.
                        qe = checkpoint_epoch(src)
                        le = checkpoint_epoch(dest)
                        if qe is None or le is None or qe <= le:
                            continue
                        items.append(PullItem(
                            rec.identity.canonical, label, src, dest, rel,
                            sst.st_size, "qnap-higher-epoch",
                            local_epoch=le, qnap_epoch=qe))
                    elif sst.st_mtime > dst.st_mtime + 2:
                        # Metadata (logs, configs, AA results): newest wins, which
                        # for a log or a sweep CSV is simply the fuller one.
                        items.append(PullItem(
                            rec.identity.canonical, label, src, dest, rel,
                            sst.st_size, "qnap-newer"))
    return items


def apply_pull(items: list[PullItem], dry_run: bool) -> int:
    """One rsync leg per (source model dir, dest model dir)."""
    legs: dict[tuple[Path, Path], list[str]] = defaultdict(list)
    # A leg carrying an epoch decision must NOT get rsync's --update; see below.
    epoch_legs: set[tuple[Path, Path]] = set()
    for it in items:
        src_dir = it.source
        for _ in range(it.rel.count("/") + 1):
            src_dir = src_dir.parent
        dest_dir = it.dest
        for _ in range(it.rel.count("/") + 1):
            dest_dir = dest_dir.parent
        legs[(src_dir, dest_dir)].append(it.rel)
        if it.reason == "qnap-higher-epoch":
            epoch_legs.add((src_dir, dest_dir))

    rc = 0
    for idx, ((src_dir, dest_dir), rels) in enumerate(
            sorted(legs.items(), key=lambda kv: str(kv[0])), 1):
        if dry_run:
            print(f"[backfill] DRY {src_dir} -> {dest_dir} ({len(rels)} files)")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".files", delete=False) as fh:
            fh.write("\n".join(rels) + "\n")
            list_path = fh.name
        try:
            cmd = list(RSYNC_BASE)
            # --update ("skip files that are newer on the receiver") as a second lock on
            # the planner's promise never to clobber a locally-newer file: the plan was
            # built from one stat() per file and applied minutes-to-hours later, and this
            # closes that window.
            #
            # NOT on a leg that carries a qnap-higher-epoch item, because there --update
            # would silently undo the epoch decision. mtime lies in these trees -- 80
            # AIRCC files read as newer on the QNAP at *identical* epochs, all carrying
            # the 2026-08-10 11:12 bulk-rewrite mtime -- which is exactly why checkpoints
            # are decided on epoch and not on time. Letting rsync re-apply an mtime rule
            # on top would drop the genuinely-more-trained checkpoint we came for.
            if (src_dir, dest_dir) not in epoch_legs:
                cmd.append("--update")
            cmd += [f"--files-from={list_path}", f"{src_dir}/", f"{dest_dir}/"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode not in (0, 23, 24):
                print(f"[backfill] ERROR rsync rc={proc.returncode}: "
                      f"{proc.stderr.strip()}", file=sys.stderr)
                rc = proc.returncode
        finally:
            os.unlink(list_path)
        if idx % 10 == 0 or idx == len(legs):
            print(f"[backfill] {_now()} {idx}/{len(legs)} model dirs pulled", flush=True)
    return rc


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="pull (default: dry run)")
    ap.add_argument("--store", type=Path, default=STORE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--min-free-gb", type=int, default=MIN_FREE_GB)
    # The weekly cron (model_store/scripts/ms_weekly_sync.sh) passes --roots qnap-slurm:
    # route 2 feeds off the Slurm archive only. The AIRCC allocation is over and its
    # archive is static, so pulling from it is a no-op that costs a full CIFS walk --
    # keep it a deliberate, by-hand choice rather than weekly work.
    ap.add_argument("--roots", nargs="+", default=list(QNAP_LABELS),
                    choices=list(QNAP_LABELS),
                    help="which QNAP archives to pull from (default: both)")
    args = ap.parse_args(argv)
    roots = tuple(dict.fromkeys(args.roots))

    if not args.store.is_dir():
        print(f"[backfill] ERROR: {args.store} does not exist -- run Step 3 first",
              file=sys.stderr)
        return 1
    for label in roots:
        if not ARCHIVE_ROOTS[label].is_dir():
            print(f"[backfill] ERROR: {ARCHIVE_ROOTS[label]} missing -- is "
                  f"{QNAP_ROOT} mounted?", file=sys.stderr)
            return 1

    print(f"[backfill] {_now()} walking {', '.join(roots)} (this takes a few minutes "
          f"over CIFS)", flush=True)
    records = build(roots=roots)
    items = plan(records, args.store, labels=roots)

    by_reason: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for it in items:
        n, b = by_reason[it.reason]
        by_reason[it.reason] = (n + 1, b + it.size)
    total = sum(it.size for it in items)
    models = len({it.canonical for it in items})

    print(f"[backfill] {_now()} plan: {len(items)} files across {models} models")
    for reason, (n, b) in sorted(by_reason.items(), key=lambda kv: -kv[1][1]):
        print(f"[backfill]   {n:6d} files  {b / GIB:9.1f} GiB  {reason}")
    print(f"[backfill]   {len(items):6d} files  {total / GIB:9.1f} GiB  TOTAL to copy")
    # 55 MB/s measured on this share (qnap_mirror.log: 52-55 MB/s sustained).
    eta_min = total / (55 * 1e6) / 60 if total else 0
    print(f"[backfill]   ETA at ~55 MB/s: {eta_min / 60:.1f} h ({eta_min:.0f} min)")

    free_gb = shutil.disk_usage(args.store).free / 1e9
    need_gb = total / 1e9
    print(f"[backfill]   free on {args.store}: {free_gb:.0f} GB, "
          f"need {need_gb:.0f} GB, floor {args.min_free_gb} GB")
    if free_gb - need_gb < args.min_free_gb:
        print(f"[backfill] REFUSING: the pull would leave "
              f"{free_gb - need_gb:.0f} GB, below the {args.min_free_gb} GB floor",
              file=sys.stderr)
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_csv = args.out_dir / "04_backfill_plan.csv"
    with plan_csv.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["model", "source_label", "source", "dest", "size", "reason",
                    "local_epoch", "qnap_epoch"])
        for it in sorted(items, key=lambda i: (i.canonical, i.rel)):
            w.writerow([it.canonical, it.source_label, it.source, it.dest,
                        it.size, it.reason,
                        "" if it.local_epoch is None else it.local_epoch,
                        "" if it.qnap_epoch is None else it.qnap_epoch])
    print(f"[backfill] wrote {plan_csv}")

    if not args.apply:
        print(f"[backfill] {_now()} DRY RUN -- nothing pulled. Re-run with --apply.")
        return 0
    return apply_pull(items, dry_run=False)


if __name__ == "__main__":
    raise SystemExit(main())
