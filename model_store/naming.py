"""Canonical model identity: name, arch, protocol, norm, eps.

Three naming conventions collide in this repo, and the whole point of this module
is that exactly one of them wins downstream:

1. **AIRCC** -- flat, arch already in the name: ``convnext_base_l2_4_init1``.
2. **Slurm ViT/Swin** -- nested, arch **only** in the parent dir:
   ``swin_b/l2_4_init1``. The leaf omits the arch, and all 31 ``swin_b`` leaf names
   are also ``vit_b_cvst`` leaf names, so keying anything on the leaf silently
   collapses the two lanes into one.
3. **Slurm convnext** -- flat like AIRCC: ``convnext_small_l2trades_2_init1``.

``canonical_name`` normalises all three to "arch is always in the name", which is
what makes a flat filename such as ``swin_b_l2_4_init1.pth.tar`` unambiguous.

The protocol vocabulary is the CSVs' own (``slurm_job_manager/csv_spec.py``
``METADATA_COLUMNS`` -> ``protocol``), so a config-derived row and a CSV-derived row
land in the same folder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Every arch that appears as a name prefix (flat convention) or a parent dir
# (nested convention). Order matters: longest prefix first, so ``convnext_base``
# is not matched by a hypothetical ``convnext`` entry.
KNOWN_ARCHES = (
    "convnext_large",
    "convnext_small",
    "convnext_base",
    "vit_b_cvst",
    "vit_m_cvst",
    "swin_b",
)

# Dirs under ``<archive>/models`` that are containers, not models. ``old_models``
# and ``vit_b_cvst_broken`` are explicitly legacy; the rest are per-arch parents.
NESTED_CONTAINERS = ("vit_b_cvst", "vit_m_cvst", "swin_b", "vit_b_cvst_broken", "old_models")

# Containers whose contents never enter the clean arch/ tree.
LEGACY_CONTAINERS = ("old_models", "vit_b_cvst_broken")

# The protocol vocabulary, as used by both csv_spec METADATA_COLUMNS.
PROTOCOLS = (
    "baseline",
    "madry",
    "trades",
    "gradnorm",
    "dvd_b_baseline",
    "v1",
    "v1_clean",
    "v1_noise",
)

NORMS = ("l1", "l2", "linf")

# Checkpoint kinds, kept in step with aircc/aircc_job_manager/best_checkpoint.py:23-27
# and aa_sweep/config.py:22-31. Do not diverge -- the DB stores one of these basenames.
CKPT_FILE_FOR_KIND = {
    "best": "model_best.pth.tar",
    "last": "last.pth.tar",
    "advbest": "model_best_adv.pth.tar",
}
KIND_FOR_CKPT_FILE = {v: k for k, v in CKPT_FILE_FOR_KIND.items()}
KEEPER_BASENAMES = frozenset(CKPT_FILE_FOR_KIND.values())

# Anchored on the leading digit so it can never match model_best* / last / epoch_*.
# ``[0-9]+`` (not ``[0-9]``) so multi-digit epochs match -- checkpoint-7,
# checkpoint-12, checkpoint-203 and checkpoint-1234 are all intermediates.
CHECKPOINT_N_RE = re.compile(r"^checkpoint-\d+\.pth\.tar$")
PERIODIC_RE = re.compile(r"^epoch_\d+\.pth\.tar$")


def is_intermediate(basename: str) -> bool:
    """True for the ``checkpoint-N.pth.tar`` files we never keep, and ``tmp.pth.tar``.

    Mirrors the rsync globs in ``backup_slurm_models.sh:214-215``
    (``--exclude='checkpoint-[0-9]*.pth.tar' --exclude='tmp.pth.tar'``); that glob
    matches any digit count, and so does this.
    """
    return basename == "tmp.pth.tar" or bool(CHECKPOINT_N_RE.match(basename))


def is_keeper_checkpoint(basename: str) -> bool:
    """True for the checkpoints the curated tree keeps."""
    return basename in KEEPER_BASENAMES or bool(PERIODIC_RE.match(basename))


def arch_from_name(name: str) -> Optional[str]:
    """Arch from a flat model name, e.g. ``convnext_base_l2_4_init1`` -> ``convnext_base``."""
    for arch in KNOWN_ARCHES:
        if name == arch or name.startswith(arch + "_"):
            return arch
    return None


def canonical_name(model_name: str) -> str:
    """Normalise a DB/dir ``model_name`` so the arch is always part of the name.

    ``swin_b/l2_4_init1``               -> ``swin_b_l2_4_init1``
    ``convnext_base_l2_4_init1``        -> unchanged (arch already present)
    ``old_models/convnext_small_l2_1_init1`` -> ``convnext_small_l2_1_init1``
    """
    parts = model_name.strip("/").split("/")
    leaf = parts[-1]
    if len(parts) == 1:
        return leaf
    parent = parts[-2]
    if arch_from_name(leaf) is not None:
        # Already self-describing (the old_models/convnext_small_* case).
        return leaf
    if parent in KNOWN_ARCHES:
        return f"{parent}_{leaf}"
    # vit_b_cvst_broken/<leaf> and anything else unexpected: keep the container in
    # the name rather than inventing an arch for it.
    return f"{parent}_{leaf}"


def flat_id(model_name: str) -> str:
    """``/``-free id matching ``aa_sweep/config.py:90``'s convention.

    Used only where a job/log name must round-trip through ``squeue -o %j``; the
    curated tree uses :func:`canonical_name` instead.
    """
    return model_name.strip("/").replace("/", "__")


@dataclass(frozen=True)
class ModelIdentity:
    """Everything needed to place a model in both trees."""

    canonical: str            # convnext_base_l2_4_init1 / swin_b_l2_4_init1
    arch: Optional[str]       # convnext_base / swin_b / ...
    protocol: Optional[str]   # madry / trades / baseline / ...
    norm: Optional[str]       # l1 / l2 / linf, or None for baselines
    eps: Optional[float]      # 1/2/4/6/8/12, or None for baselines
    init: Optional[str]
    source: str               # where the decomposition came from: csv | config | name
    legacy: bool = False      # goes to models/_legacy/ instead of models/<arch>/
    notes: str = ""
    # For legacy models, the model's path relative to ``<archive>/models``, kept
    # verbatim. Needed because the legacy buckets nest deeper than one level and
    # the leaf name alone is not unique: ``old_models/madry/convnext_small_eps-1_
    # linf_seed-1`` and ``old_models/gradnorm/convnext_small_eps-1_linf_seed-1``
    # are different models (different protocol) with the same leaf, so collapsing
    # to the leaf loses one of them outright.
    legacy_relpath: str = ""

    @property
    def experiment_relpath(self) -> Optional[str]:
        """``<arch>/<protocol>/<norm>/<canonical>.pth.tar``, norm level omitted for
        baselines. ``None`` when the model cannot be placed confidently."""
        if self.legacy or not self.arch or not self.protocol:
            return None
        parts = [self.arch, self.protocol]
        if self.norm:
            parts.append(self.norm)
        parts.append(f"{self.canonical}.pth.tar")
        return "/".join(parts)

    @property
    def store_relpath(self) -> str:
        """Where the model dir lives under ``/mnt/data4t/models``.

        Legacy models keep their original archive path under ``_legacy/`` rather
        than being renamed, which both preserves the structure that gave them
        meaning and keeps deeper nestings unambiguous.
        """
        if self.legacy:
            if self.legacy_relpath:
                return f"_legacy/{self.legacy_relpath}"
            return f"_legacy/{self.notes or 'unparsed'}/{self.canonical}"
        if not self.arch:
            return f"_legacy/unparsed/{self.canonical}"
        return f"{self.arch}/{self.canonical}"

    @property
    def record_key(self) -> str:
        """Unique key for this model across every root."""
        return f"_legacy/{self.legacy_relpath}" if (
            self.legacy and self.legacy_relpath) else self.canonical
