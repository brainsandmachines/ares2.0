"""Read a checkpoint's recorded epoch without loading its weights.

Which of two divergent copies of a model to keep is a question about *training
progress*, and mtime answers it badly: on the AIRCC side several dirs carry a
newer mtime but a far earlier epoch (199 on Slurm vs 6 on AIRCC for
``convnext_base_dvd_b_l1_1_init1``), because the run was relaunched and only got a
few epochs in before the campaign ended. Taking the newer file there would swap a
fully-trained checkpoint for a barely-started one.

The epoch is recorded inside the checkpoint, so that is what gets read. Loading it
with ``torch.load`` costs a full ~1.3 GB read per file -- ~86 GB across the
divergent set, tens of minutes on a spinning disk. But a ``torch.save`` file is a
zip archive whose ``data.pkl`` entry (~270 KB) holds the object graph, with the
tensors stored separately and referenced by persistent id. Unpickling *only* that
entry, with every torch class stubbed out and every storage reference discarded,
yields the scalar metadata -- epoch, best score, arch -- for a few hundred KB of
I/O instead of gigabytes.

Falls back to ``torch.load`` for anything that is not a zip (the legacy pre-1.6
pickle format).
"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path
from typing import Any, Optional

EPOCH_KEYS = ("epoch", "start_epoch", "Epoch")


class _Stub:
    """Stands in for any tensor-bearing class encountered while unpickling.

    Tolerates the operations pickle performs while rebuilding a graph -- item and
    attribute assignment, appends, calls -- so that stubbing one node never aborts
    the load of the scalars we actually want.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return self

    def __setattr__(self, key, value):
        pass

    def append(self, value):
        pass

    def extend(self, values):
        pass

    def update(self, *args, **kwargs):
        pass


class _MetadataUnpickler(pickle.Unpickler):
    """Unpickles the object graph while refusing to touch tensor storage."""

    def find_class(self, module: str, name: str) -> Any:
        # Only tensor-bearing / unimportable classes are stubbed. `collections` in
        # particular must NOT be: torch.save writes the checkpoint as an
        # OrderedDict, and stubbing that replaces the very mapping the metadata
        # lives in (pickle then fails on item assignment).
        if module.startswith(("torch", "numpy", "timm")):
            return _Stub
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub

    def persistent_load(self, pid: Any) -> Any:
        # Storage references -- the actual weights. Never resolved.
        return None


def _walk(obj: Any, depth: int = 0) -> Optional[int]:
    """First plausible epoch value in a nested dict, breadth-limited."""
    if depth > 3 or not isinstance(obj, dict):
        return None
    for key in EPOCH_KEYS:
        if key in obj:
            value = obj[key]
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
    for value in obj.values():
        found = _walk(value, depth + 1)
        if found is not None:
            return found
    return None


def checkpoint_epoch(path: Path) -> Optional[int]:
    """The epoch a checkpoint records, or None if it does not record one."""
    path = Path(path)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                entries = [n for n in zf.namelist() if n.endswith("data.pkl")]
                if not entries:
                    return None
                raw = zf.read(entries[0])
            return _walk(_MetadataUnpickler(io.BytesIO(raw)).load())
    except Exception:
        return None

    # Pre-1.6 checkpoints are a bare pickle; there is no cheap path, so pay for it.
    try:
        import torch
        return _walk(torch.load(path, map_location="cpu", weights_only=False))
    except Exception:
        return None
