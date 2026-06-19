"""Deterministic derivation of arch / protocol / eps / model_dir from a job name.

Single source of truth shared by the seeder and the existing-model importer, so
both classify identically and match the launcher's own parsing
(``parse_train_job`` / ``parse_cont_train_job``).
"""

from __future__ import annotations

import os
import re
from typing import Optional

ARCH_SMALL = "convnext_small"
ARCH_BASE = "convnext_base"
ARCH_LARGE = "convnext_large"
ARCHES = (ARCH_SMALL, ARCH_BASE, ARCH_LARGE)

# Allowed protocol vocabulary (descriptive; the launcher still drives off job_name).
PROTOCOLS = ("madry", "trades", "gradnorm", "v1", "dvd", "baseline")

_ARCH_PREFIX = {"convnext_base_": ARCH_BASE, "convnext_large_": ARCH_LARGE}
_CONT_EPS_RE = re.compile(r"_cont[0-9.]+to([0-9.]+)_(?:direct|ramp)_init[0-9]+$")
_STD_EPS_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)_init[0-9]+$")


def split_arch(job_name: str) -> tuple[str, str]:
    """Return (arch, job_core) — job_core is the name minus any arch prefix."""
    for prefix, arch in _ARCH_PREFIX.items():
        if job_name.startswith(prefix):
            return arch, job_name[len(prefix):]
    return ARCH_SMALL, job_name


def arch_from_name(job_name: str) -> str:
    return split_arch(job_name)[0]


def protocol_from_name(job_name: str) -> Optional[str]:
    core = split_arch(job_name)[1]
    if "gradnorm" in core:
        return "gradnorm"
    # v1 check before trades: v1clean_l2trades_* is v1, not trades
    if (core.startswith("v1clean") or core.startswith("v1noise")
            or core.startswith("v1_clean") or core.startswith("v1_noise")
            or "_v1_" in core):
        return "v1"
    if "trades" in core:
        return "trades"
    if "dvd" in core:
        return "dvd"
    if "baseline" in core:
        return "baseline"
    if any(tok in core for tok in ("linf", "l2", "l1")):
        return "madry"
    return None


def eps_from_name(job_name: str) -> Optional[float]:
    core = split_arch(job_name)[1]
    m = _CONT_EPS_RE.search(core)          # curriculum: target epsilon
    if m:
        return float(m.group(1))
    m = _STD_EPS_RE.search(core)           # standard: _<eps>_init<N>
    if m:
        return float(m.group(1))
    return None                            # baseline / clean


def model_dir_for(job_name: str, models_root: str) -> str:
    """{models_root}/convnext_<size>_<job_core>, matching the launchers.

    Raises for v1 names, whose on-disk folder embeds extra tags — those must set
    model_dir explicitly.
    """
    arch, core = split_arch(job_name)
    if any(x in core for x in ("v1clean", "v1noise", "v1_clean", "v1_noise")):
        raise ValueError(f"v1 job '{job_name}' needs an explicit model_dir")
    return os.path.join(models_root, f"{arch}_{core}")
