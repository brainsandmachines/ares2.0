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
epoch*; when it is a checkpoint of a **finished Slurm rerun** at an equal or lower
epoch whose bytes differ (see ``_completed_slurm_copy``); or when it is **metadata**
whose QNAP copy is newer. Size and mtime alone
are not enough for a checkpoint -- several QNAP-AIRCC copies are newer but far less
trained (epoch 6 against 199), so a size/mtime rule would quietly undo the epoch
decisions Step 2 gated on. ``-rt`` is preserved throughout so that mtime stays a
usable signal for every other consumer of these trees.

AutoAttack results (the three sweep CSVs, ``autoattack_eps_norm_scores.json``, the
comparison plot) are the one kind of metadata that describes a specific checkpoint,
so they travel only with it: one is pulled only when its source dir's checkpoint is
byte-identical to the one the destination will hold after this pass. Anything else is
reported as ``skip-result-other-checkpoint`` and never pulled (``_guard_results``).
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
from .hashes import HashCache, same_content
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
    reason: str        # missing | qnap-higher-epoch | qnap-slurm-rerun | qnap-newer
    local_epoch: Optional[int] = None
    qnap_epoch: Optional[int] = None
    detail: str = ""


# AutoAttack results describe the checkpoint(s) they were computed on, so they may only
# travel with them. Must stay in step with aa_sweep.config.CKPT_FILE_FOR_KIND / CSV_FOR_KIND.
KEEPER_CKPTS = ("model_best.pth.tar", "last.pth.tar", "model_best_adv.pth.tar")
RESULT_CKPTS = {
    "autoattack_sweep_results.csv": ("model_best.pth.tar",),
    "autoattack_sweep_results_last.csv": ("last.pth.tar",),
    "autoattack_sweep_results_advbest.csv": ("model_best_adv.pth.tar",),
    "autoattack_eps_norm_scores.json": KEEPER_CKPTS,
}
SKIP_OTHER_CHECKPOINT = "skip-result-other-checkpoint"


def _result_ckpts(rel: str) -> tuple[str, ...]:
    """The checkpoints a top-level AA result file describes; () if it is not one."""
    if "/" in rel:
        return ()
    if rel.startswith("autoattack_eval_comparation_") and rel.endswith(".png"):
        return KEEPER_CKPTS
    return RESULT_CKPTS.get(rel, ())


def _same_checkpoint(a: Path, b: Path, cache: HashCache) -> bool:
    """Byte-identical? Size, then the size+mtime pair the planner already trusts as
    'same file' (both routes copy with ``-t``), then sha256 -- which settles the
    bulk-rewrite case where identical bytes carry different mtimes."""
    sa, sb = a.stat(), b.stat()
    if sa.st_size != sb.st_size:
        return False
    if abs(sa.st_mtime - sb.st_mtime) < 2:
        return True
    return same_content(a, b, cache)


def _guard_results(items: list[PullItem], get_cache) -> tuple[list[PullItem], list[PullItem]]:
    """Drop every AA result pull whose source dir holds a different checkpoint than the
    one the destination will hold once this pass is applied.

    Each file is otherwise decided on its own: checkpoints on epoch, metadata on mtime.
    That let a Slurm dir's sweep CSVs -- computed on a failed first attempt -- land beside
    the AIRCC rerun's checkpoints in five store dirs (aircc_copy_audit.md, 2026-10-07),
    and re-arrive every Monday. A result whose source has no checkpoint to compare is
    dropped too when the destination holds one: it cannot be shown to describe it.
    """
    final: dict[Path, Path] = {}          # dest checkpoint -> file it will hold
    for it in items:
        if it.rel in KEEPER_CKPTS:
            final[it.dest] = it.source
    kept, skipped = [], []
    for it in items:
        ckpts = _result_ckpts(it.rel)
        mismatch = None
        for name in ckpts:
            src_ckpt = it.source.parent / name
            dest_ckpt = it.dest.parent / name
            will_hold = final.get(dest_ckpt, dest_ckpt if dest_ckpt.exists() else None)
            if will_hold is None or will_hold == src_ckpt:
                continue
            if not src_ckpt.exists():
                mismatch = f"source has no {name} to match the destination's"
                break
            if not _same_checkpoint(src_ckpt, will_hold, get_cache()):
                mismatch = f"source {name} differs from the destination's"
                break
        if mismatch:
            it.reason, it.detail = SKIP_OTHER_CHECKPOINT, mismatch
            skipped.append(it)
        else:
            kept.append(it)
    return kept, skipped


def _completed_slurm_copy(rec, src_dir: Path, dest_dir: Path) -> bool:
    """May a QNAP-Slurm checkpoint replace Botero's at an equal or LOWER epoch?

    A run reset and retrained on Slurm ends at the same final epoch (149/199) as the
    run it replaces, and its peak can land earlier -- so the higher-epoch rule never
    lets that rerun's ``last`` (or an earlier-peaking ``best``/``advbest``) reach
    Botero, while its AA results, being metadata, do. Two proofs are required that
    the QNAP copy is the finished rerun and not a snapshot taken part-way:

    * the live sjm DB says the model is ``finished`` (route 1 never copies a model
      while it is ``running``); and
    * the QNAP ``last.pth.tar`` got at least as far as Botero's, which catches a
      rerun copied while requeued that only finished after the backup.

    The frozen AIRCC root keeps the strict rule: its relaunches are the epoch-6
    copies that rule exists for.
    """
    if rec.db_source != "sjm" or rec.db_status != "finished":
        return False
    src_last, dest_last = src_dir / "last.pth.tar", dest_dir / "last.pth.tar"
    qe = checkpoint_epoch(src_last) if src_last.exists() else None
    if qe is None:
        return False
    if not dest_last.exists():
        return True
    le = checkpoint_epoch(dest_last)
    return le is not None and qe >= le


def plan(records: dict, store_root: Path,
         labels: tuple[str, ...] = QNAP_LABELS,
         cache: Optional[HashCache] = None,
         skipped: Optional[list[PullItem]] = None) -> list[PullItem]:
    """The pulls to make. AA results held back by ``_guard_results`` are appended to
    ``skipped`` (when given) so they can be reported, and never pulled."""
    items: list[PullItem] = []
    caches: list[HashCache] = [cache] if cache is not None else []
    completed: dict[tuple[Path, Path], bool] = {}
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
                        if qe is None or le is None:
                            continue
                        if qe > le:
                            items.append(PullItem(
                                rec.identity.canonical, label, src, dest, rel,
                                sst.st_size, "qnap-higher-epoch",
                                local_epoch=le, qnap_epoch=qe))
                            continue
                        # Equal or lower epoch: only a finished Slurm rerun, and only
                        # when the bytes really differ -- much of models/ is hardlinked
                        # to the old archives, and re-copying identical content would
                        # break those links for nothing.
                        if label != "qnap-slurm":
                            continue
                        key = (src_dir, dest_dir)
                        if key not in completed:
                            completed[key] = _completed_slurm_copy(rec, src_dir, dest_dir)
                        if not completed[key]:
                            continue
                        if not caches:
                            caches.append(HashCache())
                        if same_content(src, dest, caches[0]):
                            continue
                        items.append(PullItem(
                            rec.identity.canonical, label, src, dest, rel,
                            sst.st_size, "qnap-slurm-rerun",
                            local_epoch=le, qnap_epoch=qe))
                    elif sst.st_mtime > dst.st_mtime + 2:
                        # Metadata (logs, configs, AA results): newest wins, which
                        # for a log or a sweep CSV is simply the fuller one.
                        items.append(PullItem(
                            rec.identity.canonical, label, src, dest, rel,
                            sst.st_size, "qnap-newer"))

    def get_cache() -> HashCache:
        if not caches:
            caches.append(HashCache())
        return caches[0]

    kept, held = _guard_results(items, get_cache)
    if skipped is not None:
        skipped.extend(held)
    return kept


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
        if it.reason in ("qnap-higher-epoch", "qnap-slurm-rerun"):
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
            # NOT on a leg that carries a qnap-higher-epoch or qnap-slurm-rerun item
            # (both decided on epoch and content, not time), because there --update
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
    held: list[PullItem] = []
    items = plan(records, args.store, labels=roots, skipped=held)

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
    if held:
        print(f"[backfill]   {len(held):6d} AA result files NOT pulled ({SKIP_OTHER_CHECKPOINT}): "
              f"their source holds a different checkpoint than the destination will")
        for it in sorted(held, key=lambda i: (i.canonical, i.rel))[:20]:
            print(f"[backfill]     {it.canonical}/{it.rel} [{it.source_label}]: {it.detail}")
        if len(held) > 20:
            print(f"[backfill]     ... and {len(held) - 20} more (see the plan CSV)")
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
                    "local_epoch", "qnap_epoch", "detail"])
        for it in sorted(items + held, key=lambda i: (i.canonical, i.rel)):
            w.writerow([it.canonical, it.source_label, it.source, it.dest,
                        it.size, it.reason,
                        "" if it.local_epoch is None else it.local_epoch,
                        "" if it.qnap_epoch is None else it.qnap_epoch,
                        it.detail])
    print(f"[backfill] wrote {plan_csv}")

    if not args.apply:
        print(f"[backfill] {_now()} DRY RUN -- nothing pulled. Re-run with --apply.")
        return 0
    return apply_pull(items, dry_run=False)


if __name__ == "__main__":
    raise SystemExit(main())
