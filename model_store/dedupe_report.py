"""Cross-archive duplicate analysis, and the merge-decision list you approve.

Two scopes, both read-only:

* ``--scope qnap`` -- the requirement-1 report on ``/mnt/botero``. Says what is
  duplicated between the two QNAP archives and how much space that costs. The
  recommendation is deliberately "change nothing": the QNAP is the only
  off-machine copy and both writers are append-only by design.
* ``--scope data4t`` -- the requirement-2 gate. Produces the merge-decision list
  for every model that exists in both local archives, which must be approved
  before ``build_models`` writes anything.

**How a conflict is decided: highest epoch wins.** Not file age -- see ``_decide``.
An AIRCC dir can carry a newer mtime and a far earlier epoch, because the run was
relaunched there and only got a few epochs in before the campaign ended. mtime is
the tiebreak only when the epochs are equal or unreadable.

**Why hashing, not mtime.** 70 AIRCC ``model_best.pth.tar`` files share the mtime
``2026-08-10 11:12:0x`` -- written within seconds of each other by a bulk rewrite,
not by training. Three of those were confirmed byte-identical to the Slurm copy
that a newest-wins rule would have thrown away. So mtime is a pre-filter only:
same size + same mtime is *assumed* identical (cheap, and the mtime agreement is
positive evidence), while any size-equal / mtime-differing pair is hashed before
being called a conflict. ``--hash-all`` additionally verifies the assumed-identical
pairs, which is what the gate pass runs.
"""

from __future__ import annotations

import argparse
import csv as _csv
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .census import ARCHIVE_ROOTS, ModelRecord, build
from .epochs import checkpoint_epoch
from .hashes import HashCache
from .naming import KEEPER_BASENAMES, PERIODIC_RE

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "slurm_job_manager" / "logs" / "reorg"

SCOPES = {
    "qnap": ("qnap-slurm", "qnap-aircc"),
    "data4t": ("data4t-slurm", "data4t-aircc"),
}
GIB = 1024 ** 3


@dataclass
class FileVerdict:
    """One (model, checkpoint basename) pair compared across two roots."""

    canonical: str
    basename: str
    left_label: str
    right_label: str
    left: Optional[Path]
    right: Optional[Path]
    left_size: int = 0
    right_size: int = 0
    left_mtime: float = 0.0
    right_mtime: float = 0.0
    verdict: str = ""      # IDENTICAL | DIVERGENT | LEFT-ONLY | RIGHT-ONLY
    evidence: str = ""     # hash | size | mtime+size | -
    winner: str = ""       # for DIVERGENT: the label whose copy is kept
    winner_by: str = ""    # epoch | mtime (why that side won)
    gap_hours: float = 0.0
    left_epoch: Optional[int] = None
    right_epoch: Optional[int] = None
    left_aa: Optional[float] = None
    right_aa: Optional[float] = None

    @property
    def redundant_bytes(self) -> int:
        """Bytes a second copy costs, i.e. what dedup would reclaim."""
        return self.left_size if self.verdict == "IDENTICAL" else 0

    @property
    def loser_path(self) -> Optional[Path]:
        if self.verdict != "DIVERGENT":
            return None
        return self.right if self.winner == self.left_label else self.left


def _keeper_files(model_dir: Path) -> dict[str, Path]:
    """Keeper checkpoints in a model dir, keyed by the name used for comparison.

    ``periodic/epoch_NNNN.pth.tar`` keeps its ``periodic/`` prefix so a periodic
    keep never compares against a top-level checkpoint.
    """
    out: dict[str, Path] = {}
    try:
        for entry in os.scandir(model_dir):
            if entry.is_file(follow_symlinks=False) and entry.name in KEEPER_BASENAMES:
                out[entry.name] = Path(entry.path)
    except OSError:
        return out
    periodic = model_dir / "periodic"
    if periodic.is_dir():
        try:
            for entry in os.scandir(periodic):
                if entry.is_file(follow_symlinks=False) and PERIODIC_RE.match(entry.name):
                    out[f"periodic/{entry.name}"] = Path(entry.path)
        except OSError:
            pass
    return out


def compare(
    records: dict[str, ModelRecord], left_label: str, right_label: str,
    cache: HashCache, hash_all: bool, log=sys.stderr,
) -> list[FileVerdict]:
    """Classify every keeper file of every model present under both roots."""
    overlapping = sorted(
        key for key, rec in records.items()
        if left_label in rec.dirs and right_label in rec.dirs
    )
    print(f"[dedupe] {_now()} {len(overlapping)} models present under both "
          f"{left_label} and {right_label}", file=log, flush=True)

    verdicts: list[FileVerdict] = []
    for idx, key in enumerate(overlapping, 1):
        rec = records[key]
        left_files = _keeper_files(rec.dirs[left_label])
        right_files = _keeper_files(rec.dirs[right_label])
        for basename in sorted(set(left_files) | set(right_files)):
            lp, rp = left_files.get(basename), right_files.get(basename)
            fv = FileVerdict(
                canonical=rec.identity.canonical, basename=basename,
                left_label=left_label, right_label=right_label, left=lp, right=rp,
            )
            if lp is None or rp is None:
                fv.verdict = "RIGHT-ONLY" if lp is None else "LEFT-ONLY"
                present = rp if lp is None else lp
                st = present.stat()
                if lp is None:
                    fv.right_size, fv.right_mtime = st.st_size, st.st_mtime
                else:
                    fv.left_size, fv.left_mtime = st.st_size, st.st_mtime
                fv.evidence = "-"
                verdicts.append(fv)
                continue

            ls, rs = lp.stat(), rp.stat()
            fv.left_size, fv.right_size = ls.st_size, rs.st_size
            fv.left_mtime, fv.right_mtime = ls.st_mtime, rs.st_mtime
            fv.gap_hours = abs(ls.st_mtime - rs.st_mtime) / 3600.0

            if ls.st_size != rs.st_size:
                # Different size is proof of difference; no hash needed.
                fv.verdict, fv.evidence = "DIVERGENT", "size"
            elif (ls.st_dev, ls.st_ino) == (rs.st_dev, rs.st_ino):
                fv.verdict, fv.evidence = "IDENTICAL", "hardlink"
            elif abs(ls.st_mtime - rs.st_mtime) < 2 and not hash_all:
                # Same size AND same mtime: assumed identical without hashing.
                fv.verdict, fv.evidence = "IDENTICAL", "mtime+size"
            else:
                same = cache.sha256(lp, ls) == cache.sha256(rp, rs)
                fv.verdict, fv.evidence = ("IDENTICAL" if same else "DIVERGENT"), "hash"

            if fv.verdict == "DIVERGENT":
                _decide(fv, ls, rs)
            verdicts.append(fv)

        if idx % 20 == 0 or idx == len(overlapping):
            print(f"[dedupe] {_now()} {idx}/{len(overlapping)} models "
                  f"({len(verdicts)} file pairs, {len(cache)} hashes cached)",
                  file=log, flush=True)
    return verdicts


def _aa_score(path: Optional[Path], basename: str) -> Optional[float]:
    """This checkpoint kind's AutoAttack score at the model's trained threat model.

    Read from ``autoattack_eps_norm_scores.json`` beside the checkpoint, which maps
    checkpoint filename -> robust accuracy at the norm/eps the model was trained
    for. Used only to break an epoch tie.
    """
    if path is None:
        return None
    j = path.parent / "autoattack_eps_norm_scores.json"
    if not j.exists():
        return None
    try:
        scores = json.loads(j.read_text()).get("scores") or {}
    except (ValueError, OSError):
        return None
    value = scores.get(basename)
    return float(value) if isinstance(value, (int, float)) else None


def _decide(fv: FileVerdict, ls, rs) -> None:
    """Pick the surviving copy of a genuinely divergent pair.

    **Training progress, not file age.** mtime is actively misleading here: several
    AIRCC dirs carry a newer mtime but a far earlier epoch, because the run was
    relaunched there and only got a few epochs in before the campaign ended
    (``convnext_base_dvd_b_l1_1_init1`` is epoch 199 on Slurm against 6 on AIRCC).
    Keeping the newer file would throw away the trained model. So the epoch
    recorded *inside* the checkpoint decides.

    On an epoch tie, the AutoAttack score at the trained threat model breaks it --
    also evidence, and it matters: ``convnext_base_l2_4_init1``'s two
    ``last.pth.tar`` copies are both epoch 149, but the AIRCC one scores 35.84
    robust accuracy at l2/4 against the Slurm one's 0.00. mtime is the last resort.
    """
    fv.left_epoch = checkpoint_epoch(fv.left) if fv.left else None
    fv.right_epoch = checkpoint_epoch(fv.right) if fv.right else None
    le, re_ = fv.left_epoch, fv.right_epoch
    if le is not None and re_ is not None and le != re_:
        fv.winner = fv.left_label if le > re_ else fv.right_label
        fv.winner_by = "epoch"
        return
    la, ra = _aa_score(fv.left, fv.basename), _aa_score(fv.right, fv.basename)
    if la is not None and ra is not None and abs(la - ra) > 1e-9:
        fv.winner = fv.left_label if la > ra else fv.right_label
        fv.winner_by = "aa-score"
        fv.left_aa, fv.right_aa = la, ra
        return

    fv.left_aa, fv.right_aa = la, ra
    fv.winner = fv.left_label if ls.st_mtime > rs.st_mtime else fv.right_label
    fv.winner_by = "mtime"


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _date(ts: float) -> str:
    return _dt.date.fromtimestamp(ts).isoformat() if ts else "-"


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def write_csv(verdicts: list[FileVerdict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([
            "model", "file", "verdict", "evidence", "winner", "winner_by",
            "left_epoch", "right_epoch", "left_aa", "right_aa", "gap_hours",
            "left_label", "left_path", "left_size", "left_mtime",
            "right_label", "right_path", "right_size", "right_mtime",
            "loser_path",
        ])
        for v in verdicts:
            w.writerow([
                v.canonical, v.basename, v.verdict, v.evidence, v.winner,
                v.winner_by,
                "" if v.left_epoch is None else v.left_epoch,
                "" if v.right_epoch is None else v.right_epoch,
                "" if v.left_aa is None else v.left_aa,
                "" if v.right_aa is None else v.right_aa,
                f"{v.gap_hours:.1f}",
                v.left_label, v.left or "", v.left_size, _date(v.left_mtime),
                v.right_label, v.right or "", v.right_size, _date(v.right_mtime),
                v.loser_path or "",
            ])


def write_markdown(
    verdicts: list[FileVerdict], path: Path, scope: str, hash_all: bool,
    n_models_left: int, n_models_right: int,
) -> None:
    left_label, right_label = SCOPES[scope]
    identical = [v for v in verdicts if v.verdict == "IDENTICAL"]
    divergent = [v for v in verdicts if v.verdict == "DIVERGENT"]
    left_only = [v for v in verdicts if v.verdict == "LEFT-ONLY"]
    right_only = [v for v in verdicts if v.verdict == "RIGHT-ONLY"]
    models = sorted({v.canonical for v in verdicts})
    div_models = sorted({v.canonical for v in divergent})
    pure_mirrors = sorted({v.canonical for v in identical} - set(div_models))

    redundant = sum(v.redundant_bytes for v in identical)
    div_bytes = sum(v.left_size + v.right_size for v in divergent)
    loser_bytes = sum(
        (v.right_size if v.winner == v.left_label else v.left_size) for v in divergent)

    lines: list[str] = []
    add = lines.append
    add(f"# Cross-archive duplicates: `{scope}`")
    add("")
    add(f"Generated {_now()}  ·  `{left_label}` vs `{right_label}`  ·  "
        f"{'sha256 on every pair' if hash_all else 'sha256 only where mtime differs'}")
    add("")
    add("| metric | value |")
    add("|---|---|")
    add(f"| models under `{left_label}` | {n_models_left} |")
    add(f"| models under `{right_label}` | {n_models_right} |")
    add(f"| models present under **both** | {len(models)} |")
    add(f"| of those, pure mirrors (every shared file identical) | {len(pure_mirrors)} |")
    add(f"| of those, at least one divergent file | {len(div_models)} |")
    add(f"| identical file pairs | {len(identical)} |")
    add(f"| **redundant bytes (one copy is enough)** | **{redundant / GIB:.1f} GiB** |")
    add(f"| divergent file pairs | {len(divergent)} |")
    add(f"| divergent bytes, both sides | {div_bytes / GIB:.1f} GiB |")
    add(f"| divergent bytes, losers only | {loser_bytes / GIB:.1f} GiB |")
    add(f"| `{left_label}`-only files | {len(left_only)} "
        f"({sum(v.left_size for v in left_only) / GIB:.1f} GiB) |")
    add(f"| `{right_label}`-only files | {len(right_only)} "
        f"({sum(v.right_size for v in right_only) / GIB:.1f} GiB) |")
    add("")

    if divergent:
        add("## Genuine conflicts -- decisions to approve")
        add("")
        add("Only these need a decision: same filename, **verified different content**.")
        add("Everything else is either identical (one copy kept, nothing lost) or")
        add("single-sided (kept as-is).")
        add("")
        add(f"| model | file | winner | decided by | {left_label} epoch | "
            f"{right_label} epoch | evidence |")
        add("|---|---|---|---|---|---|---|")
        for v in sorted(divergent, key=lambda v: (v.canonical, v.basename)):
            add(f"| `{v.canonical}` | {v.basename} | **{v.winner}** | {v.winner_by} | "
                f"{'-' if v.left_epoch is None else v.left_epoch} | "
                f"{'-' if v.right_epoch is None else v.right_epoch} | {v.evidence} |")
        add("")
        by_mtime = [v for v in divergent if v.winner_by == "mtime"]
        if by_mtime:
            add(f"### {len(by_mtime)} decided on mtime")
            add("")
            add("Epochs tied and AutoAttack scores tied or absent, so file age was")
            add("the only signal left.")
            add("")
            for v in by_mtime:
                add(f"- `{v.canonical}` / {v.basename}: epochs "
                    f"{v.left_epoch}/{v.right_epoch}, {v.gap_hours / 24:.0f}d apart, "
                    f"winner {v.winner}")
            add("")

    if scope == "qnap":
        add("## Recommended action: change nothing")
        add("")
        add("The QNAP is the only off-machine copy, the share has ~2.2T free, and both")
        add("writers (`backup_slurm_models.sh`, `mirror_archives_to_qnap.sh`) are")
        add("append-only by design. The duplicates are the *evidence* that a model was")
        add("resumed across clusters -- deleting one side loses the provenance that makes")
        add("`models_for_experiments` auditable. The one item worth a decision is")
        add("`_pending_delete_20260810/`, staged 2026-08-10 and never confirmed.")
        add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scope", choices=sorted(SCOPES), required=True)
    ap.add_argument("--hash-all", action="store_true",
                    help="verify even the same-size/same-mtime pairs (the gate pass)")
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR)
    args = ap.parse_args(argv)

    left_label, right_label = SCOPES[args.scope]
    for label in (left_label, right_label):
        if not ARCHIVE_ROOTS[label].is_dir():
            print(f"[dedupe] ERROR: {ARCHIVE_ROOTS[label]} is not a directory "
                  f"-- is the share mounted?", file=sys.stderr)
            return 1

    print(f"[dedupe] {_now()} building census for {left_label}, {right_label}",
          file=sys.stderr, flush=True)
    records = build(roots=(left_label, right_label))
    n_left = sum(1 for r in records.values() if left_label in r.dirs)
    n_right = sum(1 for r in records.values() if right_label in r.dirs)

    cache = HashCache()
    verdicts = compare(records, left_label, right_label, cache, args.hash_all)

    prefix = "01_qnap_duplicates" if args.scope == "qnap" else "02_merge_decisions"
    if args.scope == "data4t" and not args.hash_all:
        prefix = "02_merge_decisions_mtime_only"
    md = args.out_dir / f"{prefix}.md"
    csv_path = args.out_dir / f"{prefix}.csv"
    write_markdown(verdicts, md, args.scope, args.hash_all, n_left, n_right)
    write_csv(verdicts, csv_path)

    identical = [v for v in verdicts if v.verdict == "IDENTICAL"]
    divergent = [v for v in verdicts if v.verdict == "DIVERGENT"]
    print(f"[dedupe] {_now()} done: {len(identical)} identical "
          f"({sum(v.redundant_bytes for v in identical) / GIB:.1f} GiB redundant), "
          f"{len(divergent)} divergent across "
          f"{len({v.canonical for v in divergent})} models", file=sys.stderr)
    print(f"[dedupe] wrote {md}")
    print(f"[dedupe] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
