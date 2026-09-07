"""Recover which checkpoint kind was promoted, by matching content.

``/mnt/data/robustness_models`` holds 205 bare-name checkpoints
(``convnext_small_l2_4_init1.pth.tar``) laid out ``<protocol>/<norm>/<initN>/``.
Each one is a byte copy of some archive file -- but the *kind* it came from
(``model_best.pth.tar`` vs ``last.pth.tar``) was lost when it was renamed.

That kind matters twice:

1. **convnext_small has no DB row.** Neither job DB covers it (they hold only
   convnext_base / swin_b / vit_b_cvst), and 0 of its 174 archive dirs have an
   ``autoattack_eps_norm_scores.json``. So there is no recorded ``best_checkpoint``
   to build ``models_for_experiments`` from -- except that the promotion itself
   *is* the recorded decision, made by hand and still on disk.
2. **Requirement 5 wants /mnt/data cleared**, which is only safe for files proven
   to exist in the archive. Name equality is not proof; content equality is.

So this pass hashes both sides once and answers both questions. Matching is
name-guided (a promoted ``convnext_small_l2_4_init1.pth.tar`` is compared against
the 1-3 keeper files of the archive model ``convnext_small_l2_4_init1``) with a
size-indexed fallback for the promoted files whose name matches no archive dir --
of which there are about ten, including three with a ``_model_best`` suffix that
the archive spells as a plain dir name.

Read-only.
"""

from __future__ import annotations

import argparse
import csv as _csv
import datetime as _dt
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .census import ARCHIVE_ROOTS, ModelRecord, build
from .dedupe_report import LOG_DIR, _keeper_files, _now
from .hashes import HashCache
from .naming import KIND_FOR_CKPT_FILE, canonical_name

DATA_ROOT = Path(os.environ.get("MS_DATA_ROOT", "/mnt/data"))
PROMOTED_ROOT = DATA_ROOT / "robustness_models"
GIB = 1024 ** 3

# Third-party weights, not ares output: a `robustness`-library zoo of
# resnet50/wide_resnet50 .ckpt files with no counterpart in any archive. Reported,
# never staged for deletion.
THIRD_PARTY_ROOT = DATA_ROOT / "models"


@dataclass
class Promotion:
    """One bare-name file under robustness_models, and what it turned out to be."""

    path: Path                       # /mnt/data/robustness_models/madry/l2/init1/x.pth.tar
    size: int
    name_guess: str                  # canonical model name derived from the filename
    matched_model: Optional[str] = None   # canonical name of the archive model
    matched_kind: Optional[str] = None    # best | last | advbest
    matched_path: Optional[Path] = None   # the archive file it equals
    status: str = "UNMATCHED"        # MATCHED | UNMATCHED | NO_CANDIDATES
    candidates_hashed: int = 0


def promoted_files(root: Path = PROMOTED_ROOT) -> list[Path]:
    """Every real (non-symlink) ``.pth.tar`` under robustness_models.

    ``contstim_zoo/`` is included: its 15 symlinks are skipped by the
    ``is_file(follow_symlinks=False)`` test, but the one real file in it is a
    genuine promotion and belongs in the report.
    """
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".pth.tar"):
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            out.append(path)
    return sorted(out)


def name_from_promoted(path: Path) -> str:
    """Canonical model name a promoted filename implies.

    ``convnext_small_l2_4_init1.pth.tar``              -> convnext_small_l2_4_init1
    ``convnext_small_baseline_init1_model_best.pth.tar`` -> convnext_small_baseline_init1
      (the archive spells this dir without the kind suffix; three files use it)
    """
    stem = path.name[: -len(".pth.tar")]
    for suffix in ("_model_best_adv", "_model_best", "_last"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return canonical_name(stem)


def _archive_keepers(records: dict[str, ModelRecord], labels: Iterable[str]) -> dict[str, dict[str, Path]]:
    """canonical model name -> {basename: path}, preferring the earlier label."""
    out: dict[str, dict[str, Path]] = {}
    labels = list(labels)
    for key, rec in records.items():
        merged: dict[str, Path] = {}
        for label in reversed(labels):        # later labels lose to earlier ones
            model_dir = rec.dirs.get(label)
            if model_dir:
                merged.update(_keeper_files(model_dir))
        if merged:
            out[rec.identity.canonical] = merged
    return out


def resolve(
    records: dict[str, ModelRecord], labels: Iterable[str], cache: HashCache,
    log=sys.stderr,
) -> list[Promotion]:
    keepers = _archive_keepers(records, labels)

    # Size index for the fallback: many archive files share a size, so this is
    # only consulted when the name-guided lookup fails.
    by_size: dict[int, list[Path]] = defaultdict(list)
    for files in keepers.values():
        for path in files.values():
            try:
                by_size[path.stat().st_size].append(path)
            except OSError:
                continue

    files = promoted_files()
    print(f"[promo] {_now()} {len(files)} promoted files, "
          f"{len(keepers)} archive models indexed", file=log, flush=True)

    results: list[Promotion] = []
    for idx, path in enumerate(files, 1):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        promo = Promotion(path=path, size=size, name_guess=name_from_promoted(path))

        candidates: list[tuple[Optional[str], Path]] = []
        named = keepers.get(promo.name_guess)
        if named:
            candidates = [(promo.name_guess, p) for p in named.values()]
        else:
            candidates = [(None, p) for p in by_size.get(size, [])]

        candidates = [(m, p) for m, p in candidates if _safe_size(p) == size]
        if not candidates:
            promo.status = "NO_CANDIDATES"
            results.append(promo)
            continue

        target = cache.sha256(path)
        for model, cand in candidates:
            promo.candidates_hashed += 1
            if cache.sha256(cand) == target:
                promo.status = "MATCHED"
                promo.matched_path = cand
                promo.matched_kind = KIND_FOR_CKPT_FILE.get(cand.name)
                promo.matched_model = model or _model_of(cand)
                break
        results.append(promo)

        if idx % 20 == 0 or idx == len(files):
            matched = sum(1 for r in results if r.status == "MATCHED")
            print(f"[promo] {_now()} {idx}/{len(files)} ({matched} matched, "
                  f"{len(cache)} hashes cached)", file=log, flush=True)
    return results


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _model_of(archive_file: Path) -> str:
    """Canonical model name from an archive checkpoint's path."""
    model_dir = archive_file.parent
    if model_dir.name == "periodic":
        model_dir = model_dir.parent
    for root in ARCHIVE_ROOTS.values():
        try:
            rel = model_dir.relative_to(root)
        except ValueError:
            continue
        return canonical_name(str(rel))
    return model_dir.name


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def write_reports(results: list[Promotion], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "05_promotions.csv"
    md_path = out_dir / "05_promotions.md"

    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["promoted_path", "size", "name_guess", "status",
                    "matched_model", "matched_kind", "matched_path"])
        for r in results:
            w.writerow([r.path, r.size, r.name_guess, r.status,
                        r.matched_model or "", r.matched_kind or "",
                        r.matched_path or ""])

    matched = [r for r in results if r.status == "MATCHED"]
    unmatched = [r for r in results if r.status != "MATCHED"]
    by_kind: dict[str, int] = defaultdict(int)
    for r in matched:
        by_kind[r.matched_kind or "(unknown)"] += 1
    # One kind per model, for build_experiments to consume.
    per_model: dict[str, set[str]] = defaultdict(set)
    for r in matched:
        if r.matched_model and r.matched_kind:
            per_model[r.matched_model].add(r.matched_kind)

    lines: list[str] = []
    add = lines.append
    add("# Recovered promotion decisions")
    add("")
    add(f"Generated {_now()}  ·  source `{PROMOTED_ROOT}`")
    add("")
    add("| metric | value |")
    add("|---|---|")
    add(f"| promoted files | {len(results)} ({sum(r.size for r in results) / GIB:.1f} GiB) |")
    add(f"| matched to an archive file by sha256 | {len(matched)} "
        f"({sum(r.size for r in matched) / GIB:.1f} GiB) |")
    add(f"| **not matched -- must NOT be staged for deletion** | **{len(unmatched)} "
        f"({sum(r.size for r in unmatched) / GIB:.1f} GiB)** |")
    add(f"| distinct archive models covered | {len(per_model)} |")
    add("")
    add("Promoted kind, across the matched files:")
    add("")
    for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        add(f"- `{kind}`: {count}")
    add("")

    conflicting = {m: k for m, k in per_model.items() if len(k) > 1}
    if conflicting:
        add(f"## {len(conflicting)} models promoted under more than one kind")
        add("")
        add("These have copies in several `<protocol>/<norm>/<init>/` dirs that")
        add("resolve to different checkpoints, so the promotion is ambiguous and")
        add("`build_experiments` falls through to the AA-sweep rule for them.")
        add("")
        for model, kinds in sorted(conflicting.items()):
            add(f"- `{model}`: {sorted(kinds)}")
        add("")

    if unmatched:
        add(f"## {len(unmatched)} unmatched files -- keep")
        add("")
        add("| promoted file | size | name guess | why |")
        add("|---|---|---|---|")
        for r in sorted(unmatched, key=lambda r: str(r.path)):
            why = ("no archive model of that name, and no same-size file matched"
                   if r.status == "UNMATCHED" else "no same-size candidate at all")
            add(f"| `{r.path.relative_to(DATA_ROOT)}` | {r.size / GIB:.2f} GiB | "
                f"`{r.name_guess}` | {why} |")
        add("")

    add("## Out of scope")
    add("")
    add(f"`{THIRD_PARTY_ROOT}` is a third-party `robustness`-library zoo")
    add("(`resnet50_l2_eps*.ckpt`, `wide_resnet50_4_*.ckpt`), not ares output, with no")
    add("counterpart in any archive. Reported here so it is not mistaken for a gap;")
    add("it is never staged for deletion.")
    add("")

    md_path.write_text("\n".join(lines) + "\n")
    return md_path, csv_path


def load_promoted_kinds(csv_path: Optional[Path] = None) -> dict[str, str]:
    """canonical model -> promoted kind, for ``build_experiments``.

    Models promoted under more than one kind are omitted: an ambiguous promotion
    is no decision at all, so the caller should fall through to the AA-sweep rule
    rather than pick arbitrarily.
    """
    csv_path = csv_path or (LOG_DIR / "05_promotions.csv")
    if not csv_path.exists():
        return {}
    seen: dict[str, set[str]] = defaultdict(set)
    with csv_path.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("status") != "MATCHED":
                continue
            model, kind = row.get("matched_model"), row.get("matched_kind")
            if model and kind:
                seen[model].add(kind)
    return {m: next(iter(k)) for m, k in seen.items() if len(k) == 1}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="(default) write the report")
    ap.add_argument("--roots", nargs="*", default=["data4t-slurm", "data4t-aircc"],
                    choices=sorted(ARCHIVE_ROOTS),
                    help="archive roots to match against, in preference order")
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR)
    args = ap.parse_args(argv)

    if not PROMOTED_ROOT.is_dir():
        print(f"[promo] ERROR: {PROMOTED_ROOT} is not a directory", file=sys.stderr)
        return 1

    print(f"[promo] {_now()} building census for {args.roots}", file=sys.stderr, flush=True)
    records = build(roots=args.roots)
    cache = HashCache()
    results = resolve(records, args.roots, cache)
    md, csv_path = write_reports(results, args.out_dir)

    matched = sum(1 for r in results if r.status == "MATCHED")
    print(f"[promo] {_now()} done: {matched}/{len(results)} matched")
    print(f"[promo] wrote {md}")
    print(f"[promo] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
