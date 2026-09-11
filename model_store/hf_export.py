"""Publish the zoo's EMA weights: export a local mirror of the Hugging Face repo, then upload it.

    /mnt/data4t/hf_robustness_models/                 == BrainsAndMachinesLab/robustness_models
      README.md                                       generated model card
      manifest.csv                                    one row per published model
      convnext_base/madry/l2/convnext_base_l2_4_init1/model.safetensors   state_dict_ema only
      convnext_base/madry/l2/convnext_base_l2_4_init1/metadata.json

**What gets published** is exactly ``models_for_experiments/manifest.csv``: the blessed
set, with the overrides, excludes and ``publish_as`` renames ``build_experiments`` already
applied. This module never re-decides which checkpoint is blessed.

**Only the EMA weights.** A zoo checkpoint is ~1.4 GB -- ``state_dict``,
``state_dict_ema`` and the optimizer state. Every number attached to these models refers
to the EMA copy: ``model_best`` is chosen on EMA accuracy (``adversarial_training.py``,
``eval_metrics = ema_eval_metrics``) and the AutoAttack sweeps load ``state_dict_ema``.
So that ~350 MB is what gets published, and a checkpoint without it is an error -- never
a silent fallback to ``state_dict``, which would publish weights no score describes.

**Incremental.** Each ``metadata.json`` records its source's ``hashes.file_key`` (dev,
inode, size, mtime_ns). A model whose zoo target still has that key is skipped without
opening the checkpoint. Backfill replaces a checkpoint by rsync temp+rename, which always
yields a new inode, so a replaced checkpoint is always re-exported: this relies on the
key *changing*, not on mtime meaning anything.

**The upload** is ``HfApi.upload_large_folder``: resumable (its state lives in
``<mirror>/.cache/huggingface/``), multi-worker, and it sends only what the Hub does not
already hold -- a week that added three models uploads three models.

**Nothing here deletes.** A mirror dir whose model has left the zoo is reported as an
orphan and left in place, locally and on the Hub; removing one is a deliberate hand step
(see ``model_store/README.md``).
"""

from __future__ import annotations

import argparse
import csv as _csv
import fnmatch
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .census import EXPERIMENTS_ROOT
from .dedupe_report import _now
from .hashes import CHUNK, DEFAULT_CACHE, HashCache, file_key

MIRROR_ROOT = Path(os.environ.get("MS_HF_MIRROR", "/mnt/data4t/hf_robustness_models"))
HF_REPO = os.environ.get("MS_HF_REPO", "BrainsAndMachinesLab/robustness_models")

WEIGHTS_FILE = "model.safetensors"
META_FILE = "metadata.json"
# Bump whenever metadata.json gains or changes a field. A mirror copy written under an older
# schema counts as changed, so it is re-exported instead of feeding the index a model with
# the field missing (which the card would silently render as, e.g., "clean").
METADATA_SCHEMA = 2
CKPT_SUFFIX = ".pth.tar"
# Written next to the final file and os.replace()d into place, so a killed export never
# leaves a truncated model.safetensors. Excluded from the upload.
TMP_PREFIX = ".tmp-"

# The training-config fields a user needs to rebuild, preprocess for, and attack the
# model. Deliberately not the whole hydra config: it carries cluster paths, and this repo
# is meant to go public eventually.
ATTACK_FIELDS = ("advtrain", "attack_criterion", "attack_domain", "attack_norm", "attack_eps",
                 "attack_step", "attack_it", "trades_beta", "gradnorm", "v1_attack_eps")
PREPROCESS_FIELDS = ("mean", "std", "input_size", "crop_pct", "interpolation")

MANIFEST_COLUMNS = ("name", "repo_path", "arch", "protocol", "norm", "threat_norm",
                    "eps_name", "eps_internal", "checkpoint_kind", "epoch", "best_score",
                    "parameters", "weights_sha256", "source_sha256")


class ExportError(RuntimeError):
    """A model that cannot be published as it stands (e.g. it has no EMA weights)."""


@dataclass
class Item:
    row: dict
    source: Path      # the resolved zoo target
    dest: Path        # this model's dir in the mirror
    status: str       # new | changed | unchanged | missing-source
    detail: str = ""

    @property
    def repo_path(self) -> str:
        return repo_path(self.row["relpath"])


def repo_path(relpath: str) -> str:
    """``convnext_base/madry/l2/x_init1.pth.tar`` -> ``convnext_base/madry/l2/x_init1``."""
    if not relpath.endswith(CKPT_SUFFIX):
        raise ValueError(f"zoo relpath does not end in {CKPT_SUFFIX}: {relpath!r}")
    return relpath[: -len(CKPT_SUFFIX)]


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(_csv.DictReader(fh))


def _read_meta(dest: Path) -> Optional[dict]:
    try:
        return json.loads((dest / META_FILE).read_text())
    except (OSError, ValueError):
        return None


def plan(rows: list[dict], mirror: Path, only: Optional[list[str]] = None,
         force: bool = False) -> list[Item]:
    items: list[Item] = []
    for row in rows:
        rel = row["relpath"]
        if only and not any(fnmatch.fnmatch(rel, g) for g in only):
            continue
        source = Path(os.path.realpath(row["target"]))
        dest = mirror / repo_path(rel)
        if not source.is_file():
            items.append(Item(row, source, dest, "missing-source",
                              f"zoo target {row['target']} does not exist"))
            continue
        meta = _read_meta(dest)
        weights = dest / WEIGHTS_FILE
        if meta is None:
            items.append(Item(row, source, dest, "new"))
            continue
        src_meta = meta.get("source", {})
        w_meta = meta.get("weights", {})
        if force:
            detail = "--force"
        elif meta.get("schema") != METADATA_SCHEMA:
            detail = f"metadata schema {meta.get('schema')} -> {METADATA_SCHEMA}"
        elif src_meta.get("path") != str(source):
            detail = f"blessed checkpoint moved: {src_meta.get('path')} -> {source}"
        elif src_meta.get("file_key") != file_key(source):
            detail = "source checkpoint rewritten since the last export"
        elif not weights.is_file() or weights.stat().st_size != w_meta.get("bytes"):
            detail = f"{WEIGHTS_FILE} missing or not the size its metadata records"
        else:
            items.append(Item(row, source, dest, "unchanged"))
            continue
        items.append(Item(row, source, dest, "changed", detail))
    return items


def find_orphans(mirror: Path, expected: Iterable[Path]) -> list[Path]:
    """Mirror dirs holding a model that no manifest row publishes any more."""
    if not mirror.is_dir():
        return []
    found: set[Path] = set()
    for name in (META_FILE, WEIGHTS_FILE):
        for p in mirror.rglob(name):
            if ".cache" not in p.relative_to(mirror).parts:
                found.add(p.parent)
    return sorted(found - set(expected))


# --- reading the checkpoint's training config ---------------------------------------------

def _get(obj: Any, dotted: str) -> Any:
    """``cfg["a"]["b"]`` for a DictConfig or dict, or None when any level is missing."""
    for part in dotted.split("."):
        if obj is None:
            return None
        try:
            if part not in obj:
                return None
            obj = obj[part]
        except (TypeError, KeyError, AttributeError):
            return None
    return obj


def _plain(value: Any) -> Any:
    """JSON-safe copy of a config value (DictConfig/ListConfig included)."""
    try:
        from omegaconf import Container, OmegaConf
        if isinstance(value, Container):
            try:
                return OmegaConf.to_container(value, resolve=True)
            except Exception:  # an interpolation that no longer resolves
                return OmegaConf.to_container(value, resolve=False)
    except ImportError:
        pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def threat_model(attack: dict) -> Optional[dict]:
    """The (norm, eps) a model was trained against, or None for a clean model.

    A saved checkpoint's ``attack_eps`` is already in input space: the training entrypoint
    rescales ``cfg.attacks`` in place (``normalize_attack_epsilons``) before the
    checkpoint's args are captured, so L-inf 4 is stored as 4/255 and L1 4 as 150. Both are
    recorded: ``eps_internal`` is that stored value (what the attack actually used), and
    ``eps_name`` converts it back to the units model names use (1, 2, 4, 6, 8). GradNorm
    runs train without adversarial examples (``advtrain`` false) but are scored at their
    norm/eps, so they count too.
    """
    if not (attack.get("advtrain") or attack.get("gradnorm")):
        return None
    domain = attack.get("attack_domain") or "pixel"
    norm = attack.get("attack_norm")
    stored = attack.get("attack_eps") if domain == "pixel" else attack.get("v1_attack_eps")
    if norm is None or stored is None:
        return None
    from ares.utils.epsilon_schedule import denormalize_epsilon
    return {"norm": norm, "eps_name": round(denormalize_epsilon(stored, norm, domain), 6),
            "eps_internal": stored, "domain": domain}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(TMP_PREFIX + path.name)
    tmp.write_text(text)
    os.replace(tmp, path)


def _remove_tmp(dest: Path) -> None:
    if dest.is_dir():
        for p in dest.glob(TMP_PREFIX + "*"):
            p.unlink(missing_ok=True)


# --- export --------------------------------------------------------------------------------

def build_metadata(item: Item, ckpt: dict, tensors: dict, key: str, source_sha: str,
                   weights: Path, weights_sha: str) -> dict:
    row = item.row
    cfg = _get(ckpt.get("args"), "config")
    model_cfg = _get(cfg, "model")
    builder: dict[str, Any] = {
        "arch": ckpt.get("arch"),
        "num_classes": _plain(_get(model_cfg, "num_classes")),
    }
    v1 = {str(k): _plain(model_cfg[k]) for k in (model_cfg.keys() if model_cfg is not None else ())
          if str(k).startswith("v1_")}
    if str(ckpt.get("arch", "")).endswith("_v1") and v1:
        builder["v1"] = v1
    attack = {f: _plain(_get(cfg, f"attacks.{f}")) for f in ATTACK_FIELDS}
    return {
        "schema": METADATA_SCHEMA,
        "name": row["model"],
        "repo_path": item.repo_path,
        "arch": row["arch"],
        "protocol": row["protocol"],
        "norm": row["norm"] or None,
        "checkpoint_kind": row["kind"],
        "blessing_rule": row["rule"],
        "db_source": row["db_source"],
        "best_score": float(row["best_score"]) if row.get("best_score") else None,
        "note": row.get("note") or None,
        "epoch": ckpt.get("epoch"),
        "builder": builder,
        "threat_model": threat_model(attack),
        "training": {
            "attack": attack,
            "ema_decay": _plain(_get(cfg, "training.model_ema_decay")),
        },
        "preprocessing": {f: _plain(_get(cfg, f"dataset.{f}")) for f in PREPROCESS_FIELDS},
        "weights": {
            "file": WEIGHTS_FILE,
            "tensors": len(tensors),
            "parameters": int(sum(t.numel() for t in tensors.values())),
            "dtypes": sorted({str(t.dtype).replace("torch.", "") for t in tensors.values()}),
            "bytes": weights.stat().st_size,
            "sha256": weights_sha,
        },
        "source": {
            "model_dir": item.source.parent.name,
            "checkpoint": item.source.name,
            "path": str(item.source),
            "file_key": key,
            "sha256": source_sha,
        },
        "exported_at": _now(),
    }


def export_one(item: Item, cache: HashCache) -> dict:
    import torch
    from safetensors.torch import save_file

    st = item.source.stat()
    key = file_key(item.source, st)
    source_sha = cache.sha256(item.source, st)

    ckpt = torch.load(str(item.source), map_location="cpu", mmap=True, weights_only=False)
    ema = ckpt.get("state_dict_ema") if isinstance(ckpt, dict) else None
    if not ema:
        raise ExportError(f"{item.source} has no state_dict_ema -- refusing to publish "
                          f"state_dict in its place (no score describes those weights)")
    tensors = {}
    for name, value in ema.items():
        if not torch.is_tensor(value):
            raise ExportError(f"{item.source}: state_dict_ema[{name!r}] is not a tensor")
        name = name[len("module."):] if name.startswith("module.") else name
        # clone(): off the mmap and onto its own storage, which safetensors requires.
        tensors[name] = value.detach().clone().contiguous()

    if file_key(item.source) != key:
        raise ExportError(f"{item.source} was replaced while it was being exported; re-run")

    item.dest.mkdir(parents=True, exist_ok=True)
    weights = item.dest / WEIGHTS_FILE
    tmp = item.dest / (TMP_PREFIX + WEIGHTS_FILE)
    save_file(tensors, str(tmp), metadata={"format": "pt", "name": item.row["model"]})
    os.replace(tmp, weights)
    # metadata.json last: until it lands, the recorded file_key is the old one (or absent),
    # so a run killed in between re-exports this model instead of trusting it.
    meta = build_metadata(item, ckpt, tensors, key, source_sha, weights, _sha256(weights))
    _write_text_atomic(item.dest / META_FILE, json.dumps(meta, indent=2) + "\n")
    return meta


# --- repo index: manifest.csv + README.md -------------------------------------------------

def _threat_label(tm: Optional[dict]) -> str:
    if tm is None:
        return "clean"
    eps = f"{tm['eps_name']:g}"
    label = {"linf": f"L∞ {eps}/255", "l2": f"L2 {eps}", "l1": f"L1 {eps}"}.get(
        tm["norm"], f"{tm['norm']} {eps}")
    return label if tm.get("domain", "pixel") == "pixel" else f"{label} ({tm['domain']})"


def write_index(mirror: Path, metas: list[dict], repo: str) -> None:
    metas = sorted(metas, key=lambda m: m["repo_path"])

    out = io.StringIO()
    w = _csv.writer(out, lineterminator="\n")
    w.writerow(MANIFEST_COLUMNS)
    for m in metas:
        tm = m.get("threat_model")
        w.writerow([m["name"], m["repo_path"], m["arch"], m["protocol"], m["norm"] or "",
                    tm["norm"] if tm else "", tm["eps_name"] if tm else "",
                    tm["eps_internal"] if tm else "",
                    m["checkpoint_kind"], m["epoch"],
                    "" if m["best_score"] is None else m["best_score"],
                    m["weights"]["parameters"], m["weights"]["sha256"], m["source"]["sha256"]])
    _write_text_atomic(mirror / "manifest.csv", out.getvalue())

    total_gb = sum(m["weights"]["bytes"] for m in metas) / 1e9
    card = [
        "---",
        "library_name: pytorch",
        "tags:",
        "- adversarial-robustness",
        "- image-classification",
        "- imagenet",
        "---",
        "",
        f"# {repo.split('/')[-1]}",
        "",
        f"{len(metas)} ImageNet-1k classifiers ({total_gb:.1f} GB): clean baselines and models "
        "adversarially trained with Madry PGD, TRADES and GradNorm under L1, L2 and L∞ threat "
        "models. Each folder holds the **EMA weights** of the checkpoint selected for that model "
        "-- the weights every reported score was measured on.",
        "",
        "*This file is generated by `model_store/hf_export.py` in the ares repo; edit that, "
        "not this.*",
        "",
        "## Layout",
        "",
        "```",
        "<arch>/<protocol>/[<norm>/]<name>/model.safetensors   EMA state_dict",
        "<arch>/<protocol>/[<norm>/]<name>/metadata.json       how to rebuild, preprocess and attack it",
        "manifest.csv                                          every model in one table",
        "```",
        "",
        "`metadata.json` fields worth knowing:",
        "",
        "- `builder.arch` -- the model constructor name (`builder.v1` holds the V1 front-end "
        "parameters for `*_v1` models).",
        "- `preprocessing` -- `mean`/`std`/`input_size`/`crop_pct`/`interpolation` used in "
        "training and evaluation. **Normalization is not inside the model**: feed it "
        "`(x - mean) / std` of a [0, 1] image, or wrap it in a normalizing layer so attacks "
        "operate in [0, 1] pixel space.",
        "- `threat_model` -- the norm and ε the model was trained against (`null` for clean "
        "baselines), with ε twice: `eps_name` in the units model names use (1, 2, 4, 6, 8) "
        "and `eps_internal`, the value the attack used on [0, 1] images before "
        "normalization -- L∞ `eps_internal` = `eps_name`/255, L2 `eps_internal` = "
        "`eps_name`, L1 `eps_internal` = `eps_name` × 37.5. `domain: v1_feature` (V1 models) "
        "means the training attack perturbed V1 features rather than pixels, and there L2 "
        "`eps_internal` = `eps_name` × 10. `training.attack` is the raw training config.",
        "- `checkpoint_kind` -- which saved checkpoint was selected: `best` (`model_best`), "
        "`advbest` (`model_best_adv`) or `last`. `best_score` is the score (%) the job "
        "database recorded for that choice: AutoAttack robust accuracy at the trained threat "
        "model, or clean top-1 for baselines.",
        "- `source.sha256` -- hash of the full training checkpoint these weights came from.",
        "",
        "## Loading",
        "",
        "```python",
        "import json",
        "from huggingface_hub import hf_hub_download",
        "from safetensors.torch import load_file",
        "",
        f'repo = "{repo}"',
        f'path = "{metas[0]["repo_path"] if metas else "<arch>/<protocol>/<norm>/<name>"}"',
        'meta = json.load(open(hf_hub_download(repo, f"{path}/metadata.json")))',
        'state_dict = load_file(hf_hub_download(repo, f"{path}/model.safetensors"))',
        "",
        "model = build_model(meta[\"builder\"])   # see the note below",
        "model.load_state_dict(state_dict, strict=True)",
        "```",
        "",
        "`build_model` depends on `builder.arch`:",
        "",
        "- `convnext_base`, `swin_base_patch4_window7_224` -- plain timm: "
        "`timm.create_model(arch, pretrained=False, num_classes=1000)`.",
        "- `vit_b_cvst` -- a timm model registered by the ares repo: `import ares.model.vit_convstem`, "
        "then `timm.create_model` as above.",
        "- `convnext_base_v1` -- `ares.model.v1_convnext.V1ConvNeXt(backbone_name=\"convnext_base\", ...)` "
        "with the `builder.v1` parameters (see `create_model_from_checkpoint` in "
        "`data_analysis/final_eval.py`).",
        "",
        "## Models",
    ]
    group = None
    for m in metas:
        g = (m["arch"], m["protocol"], m["norm"])
        if g != group:
            group = g
            title = " / ".join(x for x in g if x)
            card += ["", f"### {title}", "",
                     "| name | trained at | checkpoint | epoch | best_score |",
                     "|---|---|---|---|---|"]
        trained = _threat_label(m.get("threat_model"))
        score = "" if m["best_score"] is None else f"{m['best_score']:.2f}"
        card.append(f"| `{m['name']}` | {trained} | {m['checkpoint_kind']} | {m['epoch']} | {score} |")
    _write_text_atomic(mirror / "README.md", "\n".join(card) + "\n")


# --- upload --------------------------------------------------------------------------------

def upload(mirror: Path, repo: str, num_workers: Optional[int]) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as exc:  # no token, or a revoked one
        print(f"[hf] ERROR: not logged in to Hugging Face ({exc}). Run `hf auth login` "
              f"with a token that can write to {repo.split('/')[0]}.", file=sys.stderr)
        return 1
    print(f"[hf] {_now()} uploading {mirror} -> {repo} as {who.get('name')}")
    # private=True only applies if this call has to create the repo; flipping it public on
    # the Hub later is never undone by a re-run.
    api.upload_large_folder(repo_id=repo, folder_path=str(mirror), repo_type="model",
                            private=True, ignore_patterns=[TMP_PREFIX + "*", "**/" + TMP_PREFIX + "*"],
                            num_workers=num_workers)
    print(f"[hf] {_now()} upload finished")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="export into the mirror (default: dry run)")
    ap.add_argument("--upload", action="store_true",
                    help="after a clean export, upload the mirror to --repo (needs --apply)")
    ap.add_argument("--manifest", type=Path, default=EXPERIMENTS_ROOT / "manifest.csv")
    ap.add_argument("--mirror", type=Path, default=MIRROR_ROOT)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--only", nargs="*", default=None, metavar="GLOB",
                    help="export only zoo relpaths matching these fnmatch globs (* crosses /)")
    ap.add_argument("--force", action="store_true",
                    help="re-export models whose source is unchanged too (e.g. after a "
                         "metadata.json schema change)")
    ap.add_argument("--hash-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--num-workers", type=int, default=None, help="upload workers")
    args = ap.parse_args(argv)

    if args.upload and not args.apply:
        print("[hf] ERROR: --upload needs --apply", file=sys.stderr)
        return 2
    if not args.manifest.is_file():
        print(f"[hf] ERROR: {args.manifest} missing -- run the zoo pass first", file=sys.stderr)
        return 1
    rows = load_manifest(args.manifest)
    if not rows:
        print(f"[hf] ERROR: {args.manifest} has no rows -- refusing to publish nothing",
              file=sys.stderr)
        return 1

    items = plan(rows, args.mirror, args.only, force=args.force)
    orphans = find_orphans(args.mirror, (args.mirror / repo_path(r["relpath"]) for r in rows))
    by_status: dict[str, list[Item]] = {}
    for item in items:
        by_status.setdefault(item.status, []).append(item)
    counts = {s: len(by_status.get(s, [])) for s in ("new", "changed", "unchanged", "missing-source")}
    print(f"[hf] {_now()} {len(rows)} models in {args.manifest}"
          + (f", {len(items)} selected by --only" if args.only else "")
          + f": {counts}, orphans={len(orphans)}")
    for status in ("missing-source", "changed", "new"):
        for item in by_status.get(status, [])[:25]:
            print(f"[hf]   {status.upper():<14} {item.repo_path}"
                  + (f"  ({item.detail})" if item.detail else ""))
        if len(by_status.get(status, [])) > 25:
            print(f"[hf]   ... and {len(by_status[status]) - 25} more {status}")
    for o in orphans:
        print(f"[hf]   ORPHAN         {o.relative_to(args.mirror)}  (not in the zoo; left in place)")

    failed = len(by_status.get("missing-source", []))
    if not args.apply:
        print(f"[hf] {_now()} DRY RUN -- nothing written. Re-run with --apply.")
        return 1 if failed else 0

    args.mirror.mkdir(parents=True, exist_ok=True)
    cache = HashCache(args.hash_cache)
    todo = by_status.get("new", []) + by_status.get("changed", [])
    for n, item in enumerate(todo, 1):
        try:
            meta = export_one(item, cache)
        except Exception as exc:  # report every broken model, not just the first
            failed += 1
            _remove_tmp(item.dest)
            print(f"[hf] ERROR [{n}/{len(todo)}] {item.repo_path}: {exc}", file=sys.stderr)
            continue
        print(f"[hf] {_now()} [{n}/{len(todo)}] exported {item.repo_path} "
              f"({meta['weights']['bytes'] / 1e6:.1f} MB, epoch {meta['epoch']})")

    # The index lists every manifest model whose mirror copy is current -- not just this
    # run's --only selection -- so a partial export never shrinks the model card.
    current = [i for i in plan(rows, args.mirror) if i.status == "unchanged"]
    metas = [m for m in (_read_meta(i.dest) for i in current) if m is not None]
    write_index(args.mirror, metas, args.repo)
    print(f"[hf] {_now()} index written: {len(metas)} models in manifest.csv / README.md")

    if failed:
        print(f"[hf] {failed} model(s) failed -- not uploading. Fix and re-run: the export "
              f"is incremental.", file=sys.stderr)
        return 1
    if args.upload:
        return upload(args.mirror, args.repo, args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
