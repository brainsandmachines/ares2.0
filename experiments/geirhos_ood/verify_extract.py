#!/usr/bin/env python3
"""Verify already-downloaded model-vs-human tarballs and extract them safely (no network).

The tarballs are fetched by hand (see README "Data"), into ``<root>/_downloads/``. This
script then:
- checks each tarball's byte size against the GitHub release metadata saved beside it
  (the release publishes no checksums)
- records a sha256 of each tarball
- extracts it into ``<root>/datasets/`` only if every member is a plain file or directory
  under ``<task>/``, with no absolute path, no ``..`` and no links
- writes ``<root>/MANIFEST.json``

It refuses to extract over an existing ``datasets/<task>`` folder. Stdlib only, so it can
run with ``python -I``.

    python -I experiments/geirhos_ood/verify_extract.py <root> <release.json> <task> [<task> ...]
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_members(tar: tarfile.TarFile, task: str) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    for m in members:
        p = PurePosixPath(m.name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe path {m.name!r}")
        if not (m.isfile() or m.isdir()):
            raise ValueError(f"not a plain file/dir: {m.name!r} (type {m.type!r})")
        if p.parts[0] != task:
            raise ValueError(f"member outside {task}/: {m.name!r}")
    return members


def main(root: Path, release_json: Path, tasks: list[str]) -> int:
    assets = {a["name"]: a for a in json.loads(release_json.read_text())["assets"]}
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"assets": {}}
    manifest.update(source="https://github.com/bethgelab/model-vs-human/releases/tag/v0.1",
                    release_metadata=str(release_json), note="release publishes no checksums; "
                    "size checked against the GitHub API, sha256 recorded here")
    failed = 0
    for task in tasks:
        name = f"{task}.tar.gz"
        tarball = root / "_downloads" / name
        rec = {"url": assets.get(name, {}).get("browser_download_url"), "file": str(tarball),
               "expected_size": assets.get(name, {}).get("size"), "checked": dt.datetime.now().isoformat(timespec="seconds")}
        try:
            if name not in assets:
                raise ValueError("not in release metadata")
            rec["size"] = tarball.stat().st_size
            if rec["size"] != rec["expected_size"]:
                raise ValueError(f"size {rec['size']} != release {rec['expected_size']}")
            rec["sha256"] = sha256(tarball)
            dest = root / "datasets" / task
            if dest.exists():
                raise FileExistsError(f"{dest} already exists; not overwriting")
            with tarfile.open(tarball, "r:gz") as tar:
                members = check_members(tar, task)
                (root / "datasets").mkdir(exist_ok=True)
                tar.extractall(root / "datasets", members=members)
            rec["n_files"] = sum(m.isfile() for m in members)
            rec["n_png"] = sum(m.isfile() and m.name.lower().endswith(".png") for m in members)
            rec["extracted_to"] = str(dest)
            rec["status"] = "verified_extracted"
        except Exception as err:
            rec["status"] = f"failed: {err}"
            failed += 1
        manifest["assets"][name] = rec
        print(f"{task:20s} {rec['status']:20s} {rec.get('n_png', '-')} png  sha256={rec.get('sha256', '-')[:12]}")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return failed


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    sys.exit(1 if main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]) else 0)
