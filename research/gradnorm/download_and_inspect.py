#!/usr/bin/env python3
"""Download and inspect the ImageNet input-gradient-regularized checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "model_manifest.yaml"
MODELS_DIR = ROOT / "models"
EXTERNAL_DIR = ROOT / "external"
INSPECTION_DIR = ROOT / "inspection"

EXTERNAL_FILES = {
    EXTERNAL_DIR / "tulip" / "imagenet" / "resnet.py": "https://raw.githubusercontent.com/cfinlay/tulip/master/imagenet/resnet.py",
    EXTERNAL_DIR / "tulip" / "imagenet" / "fetch_model.py": "https://raw.githubusercontent.com/cfinlay/tulip/master/imagenet/fetch_model.py",
    EXTERNAL_DIR / "rig" / "eval_white_box.py": "https://raw.githubusercontent.com/adrianrm99/robustness_input_gradients/main/eval_white_box.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--inspection-dir", type=Path, default=INSPECTION_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without downloading or writing.")
    parser.add_argument("--skip-download", action="store_true", help="Inspect existing files only.")
    parser.add_argument("--skip-inspect", action="store_true", help="Download files without importing torch/model code.")
    parser.add_argument("--force", action="store_true", help="Redownload existing checkpoints and external files.")
    parser.add_argument("--keep-going", action="store_true", help="Record per-model failures and continue.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dst)


def download_google_drive(file_id: str, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except Exception as exc:
        raise RuntimeError(
            "gdown is required for the RIG Google Drive checkpoint. "
            "Run inside the project training environment or install gdown there."
        ) from exc
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(tmp), quiet=False, fuzzy=True)
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"Google Drive download failed for {file_id}")
    tmp.replace(dst)


def fetch_external_files(force: bool, dry_run: bool) -> None:
    for dst, url in EXTERNAL_FILES.items():
        if dry_run:
            print(f"[dry-run] fetch external {url} -> {dst}")
            continue
        download_url(url, dst, force=force)
        print(f"[ok] external {dst}")


def checkpoint_path(model: dict[str, Any], models_dir: Path) -> Path:
    return models_dir / model["name"] / model["checkpoint"]["filename"]


def download_checkpoint(model: dict[str, Any], models_dir: Path, force: bool, dry_run: bool) -> Path:
    ckpt = checkpoint_path(model, models_dir)
    spec = model["checkpoint"]
    if dry_run:
        source = spec.get("url") or f"gdrive:{spec.get('file_id')}"
        print(f"[dry-run] download {model['name']} from {source} -> {ckpt}")
        return ckpt
    if spec["type"] == "google_drive":
        download_google_drive(spec["file_id"], ckpt, force=force)
    elif spec["type"] == "url":
        download_url(spec["url"], ckpt, force=force)
    else:
        raise ValueError(f"Unsupported checkpoint type: {spec['type']}")
    expected = spec.get("sha256")
    actual = sha256_file(ckpt)
    if expected and actual != expected:
        bad_path = ckpt.with_suffix(ckpt.suffix + ".sha256_failed")
        if bad_path.exists():
            bad_path.unlink()
        ckpt.replace(bad_path)
        raise ValueError(f"SHA256 mismatch for {ckpt}: expected {expected}, got {actual}")
    print(f"[ok] checkpoint {ckpt} sha256={actual}")
    return ckpt


def import_tulip_resnet(external_dir: Path):
    path = external_dir / "tulip" / "imagenet" / "resnet.py"
    spec = importlib.util.spec_from_file_location("gradnorm_tulip_resnet", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import TULIP resnet from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_relu_with_gelu(module) -> None:
    import torch.nn as nn

    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU())
        else:
            replace_relu_with_gelu(child)


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    first = next(iter(state_dict))
    if not first.startswith("module."):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def unwrap_state_dict(ckpt: Any, preferred_key: str) -> tuple[dict[str, Any], str]:
    if isinstance(ckpt, dict):
        for key in (preferred_key, "state_dict_ema", "state_dict", "model", "model_state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value), key
        if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
            return strip_module_prefix(ckpt), "<root>"
    raise ValueError("Could not find a state dict in checkpoint")


def build_rig_model():
    import torch.nn as nn

    errors = []
    try:
        from timm.models import create_model

        for arch in ("tv_resnet50", "resnet50"):
            try:
                model = create_model(arch, pretrained=False, num_classes=1000)
                replace_relu_with_gelu(model)
                return model, f"timm:{arch}"
            except Exception as exc:
                errors.append(f"timm:{arch}: {exc}")
    except Exception as exc:
        errors.append(f"timm import: {exc}")

    try:
        from torchvision.models import resnet50

        model = resnet50(weights=None, num_classes=1000)
        replace_relu_with_gelu(model)
        return model, "torchvision:resnet50"
    except Exception as exc:
        errors.append(f"torchvision:resnet50: {exc}")
    raise RuntimeError("Could not build RIG ResNet50+GeLU. " + " | ".join(errors))


def build_tulip_model(external_dir: Path):
    module = import_tulip_resnet(external_dir)
    return module.resnet50(), "tulip:resnet50"


def inspect_model(model_spec: dict[str, Any], ckpt_path: Path, external_dir: Path) -> dict[str, Any]:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if model_spec["loader"] == "rig_resnet50_gelu":
        model, builder = build_rig_model()
        state_dict, state_key = unwrap_state_dict(ckpt, "state_dict_ema")
    elif model_spec["loader"] == "tulip_resnet50":
        model, builder = build_tulip_model(external_dir)
        state_dict, state_key = unwrap_state_dict(ckpt, "state_dict")
    else:
        raise ValueError(f"Unsupported loader: {model_spec['loader']}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    strict_load = not missing and not unexpected
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "name": model_spec["name"],
        "display_name": model_spec["display_name"],
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "loader": model_spec["loader"],
        "builder": builder,
        "state_key": state_key,
        "strict_load": strict_load,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "parameter_count": param_count,
        "checkpoint_top_level_keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else [],
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    print(f"[info] manifest={args.manifest}")
    if not args.skip_download:
        fetch_external_files(force=args.force, dry_run=args.dry_run)

    failures = []
    for model in manifest["models"]:
        try:
            ckpt = checkpoint_path(model, args.models_dir)
            if not args.skip_download:
                ckpt = download_checkpoint(model, args.models_dir, force=args.force, dry_run=args.dry_run)
            if args.dry_run or args.skip_inspect:
                continue
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint for inspection: {ckpt}")
            args.inspection_dir.mkdir(parents=True, exist_ok=True)
            report = inspect_model(model, ckpt, args.external_dir)
            out = args.inspection_dir / f"{model['name']}.json"
            with out.open("w") as handle:
                json.dump(report, handle, indent=2)
            status = "strict" if report["strict_load"] else "non-strict"
            print(f"[ok] inspected {model['name']} ({status}) -> {out}")
        except Exception as exc:
            if not args.keep_going:
                raise
            failure = {"name": model["name"], "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[error] {failure['name']}: {failure['error']}", file=sys.stderr)

    if failures:
        args.inspection_dir.mkdir(parents=True, exist_ok=True)
        out = args.inspection_dir / "download_or_inspection_failures.json"
        with out.open("w") as handle:
            json.dump(failures, handle, indent=2)
        raise SystemExit(f"{len(failures)} model(s) failed; see {out}")


if __name__ == "__main__":
    main()
