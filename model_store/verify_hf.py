"""Verify the Hugging Face repo against the zoo, model by model, and write the model list.

For every row of ``models_for_experiments/manifest.csv``:

* **Zoo facts** (``zoo_rows``): the linked checkpoint's epoch, the AutoAttack row for the
  blessed kind at the model's trained threat (robust + clean accuracy on the 1024 sweep
  images), the full-val ``eval_top1`` at that epoch from ``summary.csv``, and whether every
  AA row of that kind carries the checkpoint's epoch.
* **Hub check** (default, no downloads): one ``list_repo_tree`` call gives each file's LFS
  sha256 / git blob id. Required: Hub ``model.safetensors`` sha256 == the mirror file's
  sha256 == ``metadata.json`` ``weights.sha256``; Hub ``metadata.json`` blob == the mirror
  copy; its ``source.path`` is the zoo target and ``source.sha256`` the target's sha256.
  So the Hub weights were exported from exactly the checkpoint the zoo links to. Excluded
  models (``hf_excludes.csv``) must be absent; Hub model dirs no zoo row expects are extras.
* **Forward check** (``--forward``): build the model once, load the checkpoint's
  ``state_dict_ema`` (``module.`` stripped) and then the mirror ``model.safetensors``, each
  with ``strict=True``, push the same seeded batch through both, and require bitwise-equal,
  finite logits. This proves the published file *is* that checkpoint's EMA model and that it
  loads and runs.

Read-only: writes only its report CSV/markdown. Exit 1 if any check failed.

    python -m model_store.verify_hf                       # Hub check, all models
    python -m model_store.verify_hf --forward             # + forward pass (GPU, ~20 min)
    python -m model_store.verify_hf --forward --only '*swin_b*' --out-md list.md
"""

from __future__ import annotations

import argparse
import csv as _csv
import fnmatch
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from aa_sweep.config import CSV_FOR_KIND

from .build_experiments import load_excludes
from .census import EXPERIMENTS_ROOT
from .dedupe_report import LOG_DIR, _now
from .epochs import checkpoint_epoch
from .hashes import DEFAULT_CACHE, HashCache
from .hf_export import (HF_EXCLUDES_CSV, HF_REPO, META_FILE, MIRROR_ROOT, WEIGHTS_FILE,
                        excluded_reason, load_manifest, repo_path)

DEFAULT_CSV = LOG_DIR / "hf_verify.csv"
FORWARD_BATCH = 8
FORWARD_SEED = 0
# Name units: ..._l2_4_init1, ..._linftrades_cont4to8_init0, ..._dvd_b_l1_cont4to6_init1_contepoch.
EPS_IN_NAME = re.compile(r"_(?:cont\d+to)?(\d+)_init")


# --- zoo facts -----------------------------------------------------------------------------

def trained_threat(row: dict) -> Optional[tuple[str, float]]:
    """(norm, eps in name units) the model is scored at, or None for a clean model.

    The norm is the zoo's (GradNorm models are named after their penalty norm, l1, but
    trained and scored at linf); the eps is the last number before ``_init`` -- the target
    of a continuation.
    """
    norm = (row.get("norm") or "").strip()
    if not norm:
        return None
    m = EPS_IN_NAME.search(row["model"])
    return (norm, float(m.group(1))) if m else None


def _read_csv(path: Path) -> list[dict]:
    try:
        with path.open(newline="") as fh:
            return list(_csv.DictReader(fh))
    except OSError:
        return []


def _val_top1(model_dir: Path, epoch: Optional[int]) -> Optional[float]:
    if epoch is None:
        return None
    for r in _read_csv(model_dir / "summary.csv"):
        if r.get("epoch") == str(epoch) and r.get("eval_top1"):
            return round(float(r["eval_top1"]), 2)
    return None


def zoo_facts(row: dict) -> dict:
    """Everything the model list says about one zoo row, read from its store dir."""
    target = Path(os.path.realpath(row["target"]))
    epoch = checkpoint_epoch(target) if target.is_file() else None
    threat = trained_threat(row)
    aa_rows = _read_csv(target.parent / CSV_FOR_KIND[row["kind"]])
    aa_epochs = sorted({r["epoch"] for r in aa_rows})
    hit = None
    if threat:
        hits = [r for r in aa_rows if r["attack_norm"] == threat[0]
                and float(r["epsilon_input"]) == threat[1]]
        hit = hits[-1] if hits else None
    ref = hit or (aa_rows[-1] if aa_rows else None)
    return {
        "model": row["model"],
        "relpath": row["relpath"],
        "arch": row["arch"],
        "protocol": row["protocol"],
        "kind": row["kind"],
        "target": str(target),
        "epoch": epoch,
        "threat": f"{threat[0]}@{threat[1]:g}" if threat else "clean",
        "clean_aa1k": round(float(ref["clean_acc"]), 2) if ref else None,
        "val_top1": _val_top1(target.parent, epoch),
        "aa": round(float(hit["robust_acc"]), 2) if hit else None,
        "aa_cells": len({(r["attack_norm"], r["epsilon_input"]) for r in aa_rows}),
        # every AA row of the blessed kind was computed on this very checkpoint's epoch
        "aa_epoch_ok": bool(aa_rows) and aa_epochs == [str(epoch)],
        "best_score": row.get("best_score") or None,
    }


def zoo_rows(rows: Iterable[dict]) -> list[dict]:
    return [zoo_facts(r) for r in rows]


# --- Hub check -----------------------------------------------------------------------------

def git_blob_sha1(path: Path) -> str:
    """The git blob id the Hub reports for a small (non-LFS) file."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hub_index(api: Any, repo: str) -> dict[str, dict]:
    """repo file path -> {"sha256": LFS sha256 or None, "blob_id": git blob id}."""
    out: dict[str, dict] = {}
    for f in api.list_repo_tree(repo, recursive=True):
        if not hasattr(f, "blob_id"):  # a folder
            continue
        lfs = getattr(f, "lfs", None)
        out[f.path] = {"sha256": getattr(lfs, "sha256", None) if lfs else None,
                       "blob_id": f.blob_id}
    return out


def hub_model_dirs(index: dict[str, dict]) -> set[str]:
    return {p.rsplit("/", 1)[0] for p in index if p.endswith("/" + WEIGHTS_FILE)
            or p.endswith("/" + META_FILE)}


def check_hub(fact: dict, mirror: Path, index: dict[str, dict], cache: HashCache,
              excluded: Optional[str]) -> dict:
    """Hub status and sha checks for one model; ``problems`` is empty when all pass."""
    rp = repo_path(fact["relpath"])
    on_hub = f"{rp}/{WEIGHTS_FILE}" in index or f"{rp}/{META_FILE}" in index
    out = {"hf_status": "", "hub_weights_sha256": "", "problems": []}
    if excluded is not None:
        out["hf_status"] = "excluded"
        if on_hub:
            out["hf_status"] = "excluded-but-on-hub"
            out["problems"].append("excluded model is on the Hub")
        return out
    if not on_hub:
        out["hf_status"] = "not-uploaded"
        return out
    out["hf_status"] = "on-hub"
    problems = out["problems"]
    hub_w = index.get(f"{rp}/{WEIGHTS_FILE}")
    hub_m = index.get(f"{rp}/{META_FILE}")
    dest = mirror / rp
    if hub_w is None or hub_m is None:
        problems.append("Hub has only one of model.safetensors / metadata.json")
        return out
    out["hub_weights_sha256"] = hub_w["sha256"] or ""
    weights, meta_path = dest / WEIGHTS_FILE, dest / META_FILE
    if not weights.is_file() or not meta_path.is_file():
        problems.append("model is on the Hub but missing from the local mirror")
        return out
    meta = json.loads(meta_path.read_text())
    if hub_w["sha256"] != meta["weights"]["sha256"]:
        problems.append("Hub weights sha256 != metadata weights.sha256")
    if cache.sha256(weights) != hub_w["sha256"]:
        problems.append("Hub weights sha256 != mirror model.safetensors sha256")
    if git_blob_sha1(meta_path) != hub_m["blob_id"]:
        problems.append("Hub metadata.json differs from the mirror copy")
    if meta["source"]["path"] != fact["target"]:
        problems.append(f"exported from {meta['source']['path']}, zoo links {fact['target']}")
    elif cache.sha256(Path(fact["target"])) != meta["source"]["sha256"]:
        problems.append("zoo checkpoint sha256 != metadata source.sha256 (re-export needed)")
    if meta.get("epoch") != fact["epoch"] or meta.get("checkpoint_kind") != fact["kind"]:
        problems.append("metadata epoch/kind disagree with the zoo checkpoint")
    return out


# --- forward check -------------------------------------------------------------------------

def build_model(meta: dict, ckpt_args: Any):
    """The architecture ``metadata.builder`` names, as the model card documents it."""
    builder = meta["builder"]
    arch, num_classes = builder["arch"], builder.get("num_classes") or 1000
    if arch.startswith("convnext_") and arch.endswith("_v1"):
        from types import SimpleNamespace
        from data_analysis.final_eval import create_model_from_checkpoint
        size = (meta.get("preprocessing") or {}).get("input_size") or 224
        return create_model_from_checkpoint(
            arch, ckpt_args, SimpleNamespace(num_classes=num_classes, input_size=size))
    import timm
    import ares.model.vit_convstem  # noqa: F401  registers vit_b_cvst with timm
    return timm.create_model(arch, pretrained=False, num_classes=num_classes)


def forward_check(fact: dict, mirror: Path, device: str) -> tuple[bool, str]:
    """Same seeded batch through the checkpoint's EMA weights and the published file."""
    import torch
    from safetensors.torch import load_file

    dest = mirror / repo_path(fact["relpath"])
    meta = json.loads((dest / META_FILE).read_text())
    ckpt = torch.load(fact["target"], map_location="cpu", mmap=True, weights_only=False)
    ema = {(k[len("module."):] if k.startswith("module.") else k): v
           for k, v in ckpt["state_dict_ema"].items()}
    published = load_file(str(dest / WEIGHTS_FILE))
    model = build_model(meta, ckpt.get("args")).to(device).eval()
    size = (meta.get("preprocessing") or {}).get("input_size") or 224
    x = torch.randn(FORWARD_BATCH, 3, size, size,
                    generator=torch.Generator().manual_seed(FORWARD_SEED)).to(device)

    def logits(state_dict: dict) -> "torch.Tensor":
        model.load_state_dict(state_dict, strict=True)
        torch.manual_seed(FORWARD_SEED)  # any stochastic layer draws the same noise
        with torch.no_grad():
            return model(x).float().cpu()

    try:
        a, b = logits(ema), logits(published)
    finally:
        del model, ckpt, ema, published
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not torch.isfinite(a).all():
        return False, "non-finite logits"
    if not torch.equal(a, b):
        return False, f"logits differ (max |diff| {(a - b).abs().max().item():.3g})"
    return True, f"top1 agree {int((a.argmax(1) == b.argmax(1)).sum())}/{len(a)}"


# --- report --------------------------------------------------------------------------------

REPORT_COLUMNS = ("model", "relpath", "arch", "protocol", "threat", "kind", "epoch",
                  "clean_aa1k", "val_top1", "aa", "aa_cells", "aa_epoch_ok", "best_score",
                  "hf_status", "hub_weights_sha256", "forward", "forward_detail", "problems",
                  "target")


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None or v == "":
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def markdown_table(facts: list[dict]) -> str:
    lines = ["| # | model | threat | ckpt | ep | clean (AA 1k) | val top1 (50k) | AA @ threat "
             "| HF | Hub weights sha256 | checks |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, f in enumerate(facts, 1):
        checks = []
        if f.get("aa_epoch_ok"):
            checks.append("AA epoch ✅")
        elif f["threat"] != "clean":
            checks.append("AA epoch ❌")
        if f.get("hf_status") == "on-hub":
            checks.append("sha ❌" if f.get("problems") else "sha ✅")
        if f.get("forward") in (True, False):
            checks.append("forward ✅" if f["forward"] else "forward ❌")
        aa = "—" if f["threat"] == "clean" else _fmt(f["aa"])
        sha = (f.get("hub_weights_sha256") or "")[:12]
        lines.append(f"| {n} | {f['model']} | {f['threat']} | {f['kind']} | {_fmt(f['epoch'])} "
                     f"| {_fmt(f['clean_aa1k'])} | {_fmt(f['val_top1'])} | {aa} "
                     f"| {f.get('hf_status', '')} | {sha} | {', '.join(checks)} |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, facts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for f in facts:
            w.writerow({**f, "problems": "; ".join(f.get("problems", []))})
    os.replace(tmp, path)


# --- main ----------------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=EXPERIMENTS_ROOT / "manifest.csv")
    ap.add_argument("--mirror", type=Path, default=MIRROR_ROOT)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--excludes", type=Path, default=HF_EXCLUDES_CSV)
    ap.add_argument("--hash-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--only", nargs="*", default=None, metavar="GLOB",
                    help="check only zoo relpaths matching these fnmatch globs")
    ap.add_argument("--forward", action="store_true",
                    help="also run the checkpoint-vs-published forward pass for on-Hub models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-md", type=Path, default=None, help="also write the model list here")
    args = ap.parse_args(argv)

    rows = load_manifest(args.manifest)
    if args.only:
        rows = [r for r in rows if any(fnmatch.fnmatch(r["relpath"], g) for g in args.only)]
    excludes = load_excludes(args.excludes)
    facts = zoo_rows(rows)

    from huggingface_hub import HfApi
    index = hub_index(HfApi(), args.repo)
    cache = HashCache(args.hash_cache)
    for f in facts:
        f.update(check_hub(f, args.mirror, index, cache, excluded_reason(f["model"], excludes)))
        if f["threat"] != "clean" and not f["aa_epoch_ok"]:
            f["problems"].append("AA rows are not all from the linked checkpoint's epoch")
        if f["threat"] != "clean" and f["aa"] is None:
            f["problems"].append("no AA result at the trained threat")

    # Hub dirs no zoo row accounts for (an excluded model still on the Hub is already a
    # per-model problem above).
    known = {repo_path(r["relpath"]) for r in load_manifest(args.manifest)}
    extras = [] if args.only else sorted(hub_model_dirs(index) - known)

    if args.forward:
        todo = [f for f in facts if f["hf_status"] == "on-hub"]
        for n, f in enumerate(todo, 1):
            try:
                ok, detail = forward_check(f, args.mirror, args.device)
            except Exception as exc:  # report every model, not just the first broken one
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            f["forward"], f["forward_detail"] = ok, detail
            if not ok:
                f["problems"].append(f"forward: {detail}")
            print(f"[verify] {_now()} [{n}/{len(todo)}] {f['model']}: "
                  f"{'ok' if ok else 'FAIL'} ({detail})", flush=True)

    write_csv(args.out_csv, facts)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown_table(facts), encoding="utf-8")

    bad = [f for f in facts if f["problems"]]
    counts: dict[str, int] = {}
    for f in facts:
        counts[f["hf_status"]] = counts.get(f["hf_status"], 0) + 1
    print(f"[verify] {_now()} {len(facts)} zoo models: {counts}; "
          f"{len(bad)} with problems; {len(extras)} unexpected Hub dirs"
          + (f"; forward checked {sum(1 for f in facts if 'forward' in f)}" if args.forward else ""))
    for f in bad:
        print(f"[verify]   PROBLEM {f['model']}: {'; '.join(f['problems'])}")
    for e in extras:
        print(f"[verify]   EXTRA   {e}  (on the Hub, not expected by the zoo)")
    print(f"[verify] report: {args.out_csv}" + (f", {args.out_md}" if args.out_md else ""))
    return 1 if bad or extras else 0


if __name__ == "__main__":
    sys.exit(main())
