"""Weekly Hugging Face upload proposals: mail what is ready, upload only what was approved.

Route 2 (``ms_weekly_sync.sh``, Monday) rebuilds the zoo and then runs ``propose``. If the
zoo holds models whose HF export is missing or stale (``hf_export.plan`` status ``new`` /
``changed``, ``hf_excludes.csv`` honoured), it writes proposal **#N** and mails it:

    slurm_job_manager/logs/reorg/hf_proposals/HF-<N>.csv    one row per model, frozen facts
    slurm_job_manager/logs/reorg/hf_proposals/HF-<N>.json   status: open | superseded | uploaded | failed

Each row pins the zoo target's path, ``file_key`` and sha256 at proposal time, plus what the
model list shows (epoch, clean, val top1, AA at the trained threat) and any flag worth a
second look (AA < 10, AA missing, AA rows from another epoch).

Nothing is uploaded until the user says **"approved mail #N"** (in a Claude Code session),
which runs ``approve N``:

1. the proposal must be open, or failed on an earlier try (a newer proposal supersedes an
   older open one);
2. every row's zoo target must still be the same file (path, file_key, sha256) -- if a
   checkpoint moved since the mail, the approval no longer describes it and nothing uploads;
3. ``hf_export --apply --upload --only <exactly those models>``;
4. ``verify_hf --forward --only <those models>``; the proposal is marked uploaded or failed.

    python -m model_store.hf_proposal propose [--dry-run]   # cron, after the weekly sync
    python -m model_store.hf_proposal status
    python -m model_store.hf_proposal approve 3
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .build_experiments import load_excludes
from .census import EXPERIMENTS_ROOT
from .dedupe_report import LOG_DIR, _now
from .hashes import DEFAULT_CACHE, HashCache, file_key
from .hf_export import HF_EXCLUDES_CSV, HF_REPO, MIRROR_ROOT, load_manifest, plan
from .verify_hf import zoo_facts

PROPOSAL_DIR = Path(os.environ.get("MS_HF_PROPOSALS", LOG_DIR / "hf_proposals"))
LOW_AA = 10.0
COLUMNS = ("model", "relpath", "reason", "threat", "kind", "epoch", "clean_aa1k", "val_top1",
           "aa", "flags", "target", "file_key", "sha256")


# --- proposal files ------------------------------------------------------------------------

def _paths(pdir: Path, n: int) -> tuple[Path, Path]:
    return pdir / f"HF-{n}.csv", pdir / f"HF-{n}.json"


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def load_status(pdir: Path, n: int) -> dict:
    return json.loads(_paths(pdir, n)[1].read_text())


def load_rows(pdir: Path, n: int) -> list[dict]:
    with _paths(pdir, n)[0].open(newline="") as fh:
        return list(_csv.DictReader(fh))


def all_ids(pdir: Path) -> list[int]:
    out = []
    for p in pdir.glob("HF-*.json"):
        try:
            out.append(int(p.stem.split("-", 1)[1]))
        except ValueError:
            continue
    return sorted(out)


def _flags(fact: dict) -> str:
    flags = []
    if fact["threat"] != "clean":
        if fact["aa"] is None:
            flags.append("AA missing at trained threat")
        elif fact["aa"] < LOW_AA:
            flags.append(f"low AA (<{LOW_AA:g})")
        if fact["aa"] is not None and not fact["aa_epoch_ok"]:
            flags.append("AA rows from another epoch")
    return "; ".join(flags)


def candidates(manifest: Path, mirror: Path, excludes_csv: Path) -> list:
    """hf_export plan items whose Hub copy is missing or stale."""
    items = plan(load_manifest(manifest), mirror, excludes=load_excludes(excludes_csv))
    return [i for i in items if i.status in ("new", "changed")]


def proposal_rows(items: list, cache: HashCache) -> list[dict]:
    rows = []
    for item in items:
        fact = zoo_facts(item.row)
        src = Path(fact["target"])
        rows.append({
            **{k: fact[k] for k in ("model", "relpath", "threat", "kind", "epoch",
                                     "clean_aa1k", "val_top1", "aa", "target")},
            "reason": item.status + (f" ({item.detail})" if item.detail else ""),
            "flags": _flags(fact),
            "file_key": file_key(src),
            "sha256": cache.sha256(src),
        })
    return rows


def _cell(v) -> str:
    return "n/a" if v in (None, "") else str(v)


def mail_body(n: int, rows: list[dict], superseded: list[int]) -> str:
    lines = [f"HF upload proposal #{n} -- {len(rows)} zoo model(s) are not on Hugging Face "
             f"({HF_REPO}) or have a stale export.", ""]
    head = ("model", "threat", "ckpt", "epoch", "clean(AA1k)", "val top1", "AA", "flags")
    table = [head] + [(r["model"], r["threat"], r["kind"], _cell(r["epoch"]), _cell(r["clean_aa1k"]),
                       _cell(r["val_top1"]), _cell(r["aa"]) if r["threat"] != "clean" else "-",
                       r["flags"] or "") for r in rows]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(head))]
    for row in table:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    lines += ["", "Checkpoints are pinned by path + sha256; if one changes before approval, the",
              "approval is refused and next Monday's proposal lists it again.", ""]
    if superseded:
        lines.append("This supersedes open proposal(s): " + ", ".join(f"#{s}" for s in superseded))
        lines.append("")
    lines += [f'To upload exactly these models, tell Claude Code:  approved mail #{n}',
              f"(runs: python -m model_store.hf_proposal approve {n})",
              f"Details: {PROPOSAL_DIR}/HF-{n}.csv"]
    return "\n".join(lines) + "\n"


# --- commands ------------------------------------------------------------------------------

def cmd_propose(args) -> int:
    items = candidates(args.manifest, args.mirror, args.excludes)
    if not items:
        print(f"[hf-proposal] {_now()} nothing to propose: every non-excluded zoo model is on the Hub")
        return 0
    rows = proposal_rows(items, HashCache(args.hash_cache))
    pdir: Path = args.proposals
    ids = all_ids(pdir) if pdir.is_dir() else []
    n = (ids[-1] + 1) if ids else 1
    open_ids = [i for i in ids if load_status(pdir, i)["status"] == "open"]
    subject = f"[ares] HF upload proposal #{n}: {len(rows)} model(s)"
    body = mail_body(n, rows, open_ids)
    if args.dry_run:
        print(subject + "\n\n" + body)
        print(f"[hf-proposal] DRY RUN -- nothing written, no mail (would be #{n})")
        return 0

    pdir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = _paths(pdir, n)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)
    _write_json(json_path, {"id": n, "created": _now(), "status": "open",
                            "models": [r["model"] for r in rows]})
    for i in open_ids:
        st = load_status(pdir, i)
        st.update(status="superseded", superseded_by=n, superseded_at=_now())
        _write_json(_paths(pdir, i)[1], st)

    from aircc.aircc_job_manager.notify import make_emailer
    emailer = make_emailer(source="model_store.hf")
    if emailer is None:
        print(f"[hf-proposal] WARNING: no mail transport configured -- proposal #{n} written "
              f"to {csv_path} but not mailed", file=sys.stderr)
        print(subject + "\n\n" + body)
        return 1
    emailer(subject, body, urgent=True)
    print(f"[hf-proposal] {_now()} proposal #{n} ({len(rows)} models) written and mailed")
    return 0


def check_pins(rows: list[dict], manifest: Path, cache: HashCache) -> list[str]:
    """Why each proposed model no longer matches the zoo (empty = still exactly as mailed)."""
    zoo = {r["model"]: r for r in load_manifest(manifest)}
    problems = []
    for r in rows:
        z = zoo.get(r["model"])
        if z is None:
            problems.append(f"{r['model']}: no longer in the zoo")
            continue
        target = Path(os.path.realpath(z["target"]))
        if str(target) != r["target"]:
            problems.append(f"{r['model']}: zoo now links {target}, proposal pinned {r['target']}")
        elif not target.is_file() or file_key(target) != r["file_key"]:
            if not target.is_file() or cache.sha256(target) != r["sha256"]:
                problems.append(f"{r['model']}: checkpoint changed since the proposal")
    return problems


def cmd_approve(args) -> int:
    pdir: Path = args.proposals
    n = args.id
    if not _paths(pdir, n)[1].exists():
        print(f"[hf-proposal] ERROR: no proposal #{n} in {pdir}", file=sys.stderr)
        return 2
    st = load_status(pdir, n)
    if st["status"] not in ("open", "failed"):  # a failed upload may be approved again
        print(f"[hf-proposal] ERROR: proposal #{n} is {st['status']}"
              + (f" (by #{st['superseded_by']})" if st.get("superseded_by") else "")
              + " -- only an open (or failed) proposal can be approved", file=sys.stderr)
        return 2
    rows = load_rows(pdir, n)
    problems = check_pins(rows, args.manifest, HashCache(args.hash_cache))
    if problems:
        for p in problems:
            print(f"[hf-proposal] REFUSED: {p}", file=sys.stderr)
        print(f"[hf-proposal] nothing uploaded; next Monday's proposal will list the current "
              f"checkpoints", file=sys.stderr)
        return 1

    from . import hf_export, verify_hf
    relpaths = [r["relpath"] for r in rows]
    st.update(status="approved", approved_at=_now())
    _write_json(_paths(pdir, n)[1], st)
    common = ["--manifest", str(args.manifest), "--mirror", str(args.mirror),
              "--excludes", str(args.excludes), "--hash-cache", str(args.hash_cache)]
    rc = hf_export.main(["--apply", "--upload", *common, "--only", *relpaths])
    if rc == 0:
        rc = verify_hf.main(["--forward", *common, "--device", args.device,
                             "--out-csv", str(pdir / f"HF-{n}.verify.csv"), "--only", *relpaths])
    st.update(status="uploaded" if rc == 0 else "failed", finished_at=_now(), rc=rc)
    _write_json(_paths(pdir, n)[1], st)
    print(f"[hf-proposal] {_now()} proposal #{n}: {st['status']} (rc={rc})")
    return rc


def cmd_status(args) -> int:
    pdir: Path = args.proposals
    ids = all_ids(pdir) if pdir.is_dir() else []
    if not ids:
        print("[hf-proposal] no proposals yet")
    for i in ids:
        st = load_status(pdir, i)
        print(f"#{i}  {st['status']:<10}  {st['created']}  {len(st['models'])} models"
              + (f"  (superseded by #{st['superseded_by']})" if st.get("superseded_by") else ""))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=EXPERIMENTS_ROOT / "manifest.csv")
    ap.add_argument("--mirror", type=Path, default=MIRROR_ROOT)
    ap.add_argument("--excludes", type=Path, default=HF_EXCLUDES_CSV)
    ap.add_argument("--hash-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--proposals", type=Path, default=PROPOSAL_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="write + mail proposal #N if anything awaits upload")
    p.add_argument("--dry-run", action="store_true", help="print the mail; write and send nothing")
    a = sub.add_parser("approve", help="upload exactly the models of open proposal #N")
    a.add_argument("id", type=int)
    a.add_argument("--device", default="cuda", help="for the post-upload forward check")
    sub.add_parser("status", help="list proposals")
    args = ap.parse_args(argv)
    return {"propose": cmd_propose, "approve": cmd_approve, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
