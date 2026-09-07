"""Decompose a model from its own on-disk config, for runs that are in no CSV.

The job-manager CSVs carry ``arch / protocol / threat_norm / threat_eps / init``
for every model they launched, but they only cover the *current* campaigns:
``convnext_base`` (AIRCC) and ``vit_b_cvst`` / ``swin_b`` (Slurm). The 174
``convnext_small`` dirs in the Slurm archive predate them and appear in no CSV at
all, as do five hand-launched ``*_pgd5*`` AIRCC runs.

Every one of those dirs does carry its own config -- verified: 174/174 have at
least one of the three shapes below, none missing. So the decomposition is
recoverable without guessing from the folder name.

Two generations, and they nest differently:

* **old gen** ``args.yaml`` (144 dirs) -- flat argparse dump. ``model``,
  ``attack_norm``, ``attack_eps``, ``attack_criterion``, ``gradnorm``,
  ``trades_beta``, and ``output_dir`` (whose basename is the model name, because
  ``experiment_name`` is ``null`` in this generation).
* **new gen** ``hydra_config.yaml`` / ``runtime_config.yaml`` (122 dirs) -- nested
  Hydra config. ``model.model``, ``model.experiment_name``,
  ``attacks.attack_norm``, ``attacks.attack_eps``, ``attacks.attack_criterion``,
  ``attacks.advtrain``, ``attacks.gradnorm``, ``dataset.dvd.{enabled,variant}``,
  ``model.v1_noise_mode``.
* **fallback** ``.hydra/config.yaml`` (167 dirs) -- same nesting as new gen.

The folder name is used **only as a cross-check**. Where the config and the name
disagree the model is reported and sent to ``models/_legacy/unparsed/`` rather than
being filed into a protocol folder on a guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .naming import NORMS, ModelIdentity, arch_from_name, canonical_name

# Probed in order; the first that parses wins.
CONFIG_CANDIDATES = (
    "hydra_config.yaml",
    "runtime_config.yaml",
    ".hydra/config.yaml",
    "args.yaml",
)


def _load_yaml(path: Path) -> Optional[dict]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a hard dep of the repo
        return None
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _dig(cfg: dict, dotted: str) -> Any:
    """``_dig(cfg, "attacks.attack_eps")`` -- tolerant of a missing level."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first(cfg: dict, *dotted: str) -> Any:
    """First non-None value among several candidate keys.

    Lets one call cover both generations: ``_first(cfg, "attacks.attack_eps",
    "attack_eps")`` reads the nested new-gen key or the flat old-gen key.
    """
    for key in dotted:
        val = _dig(cfg, key)
        if val is not None:
            return val
    return None


def _as_float(val: Any) -> Optional[float]:
    if val is None or val == "" or (isinstance(val, str) and val.strip().lower() in ("none", "null")):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes")


def _norm_of(val: Any) -> Optional[str]:
    if val is None:
        return None
    text = str(val).strip().lower()
    return text if text in NORMS else None


def _protocol_of(cfg: dict, eps: Optional[float], name: str) -> Optional[str]:
    """Map a config onto the CSVs' own protocol vocabulary.

    Order matters: the V1 front-end and GradNorm are orthogonal to madry/trades and
    have their own protocol names in the CSVs, so they are tested first. DVD is a
    data-augmentation prefix and only ever pairs with a *baseline* protocol name in
    the CSV vocabulary (``dvd_b_baseline``); a DVD run that also trains
    adversarially keeps its madry/trades protocol, matching how the CSVs name it.
    """
    arch_model = str(_first(cfg, "model.model", "model") or "")
    v1_noise = _first(cfg, "model.v1_noise_mode", "v1_noise_mode")
    v1_noise_set = v1_noise is not None and str(v1_noise).strip().lower() not in ("none", "null", "")
    is_v1 = arch_model.endswith("_v1") or "_v1_" in f"_{name}_"

    adv = _first(cfg, "attacks.advtrain", "advtrain")
    advtrain = _as_bool(adv) if adv is not None else (eps is not None and eps > 0)
    criterion = str(_first(cfg, "attacks.attack_criterion", "attack_criterion") or "").strip().lower()
    gradnorm = _as_bool(_first(cfg, "attacks.gradnorm", "gradnorm"))
    dvd_enabled = _as_bool(_dig(cfg, "dataset.dvd.enabled"))

    if is_v1:
        if v1_noise_set:
            return "v1_noise"
        if not advtrain or eps in (None, 0):
            return "v1_clean"
        return "v1"
    if gradnorm:
        return "gradnorm"
    if not advtrain or eps in (None, 0):
        return "dvd_b_baseline" if dvd_enabled else "baseline"
    if criterion == "trades":
        return "trades"
    if criterion:
        # mixup / ce / anything else = plain adversarial training.
        return "madry"
    return None


def read_identity(model_dir: Path, fallback_name: Optional[str] = None) -> Optional[ModelIdentity]:
    """Decompose ``model_dir`` from its own config, or ``None`` if no config parses.

    ``fallback_name`` is the name derived from the directory path; it is used when
    the config carries no ``experiment_name`` (old gen) and as the cross-check
    otherwise.
    """
    cfg: Optional[dict] = None
    src_file = ""
    for candidate in CONFIG_CANDIDATES:
        path = model_dir / candidate
        if path.exists():
            cfg = _load_yaml(path)
            if cfg:
                src_file = candidate
                break
    if not cfg:
        return None

    # Name: the DIRECTORY is authoritative. Both job managers lay a run out as
    # ``<models_root>/<model_name>``, so the dir path *is* the identity, and the
    # config only corroborates it.
    #
    # This ordering matters: some ``.hydra/config.yaml`` files carry a *relative*
    # ``output_dir`` of ``results/models``, whose basename is "models". Trusting the
    # config first renamed such a model to ``models`` and filed it as
    # ``convnext_small/gradnorm/linf/models.pth.tar`` -- silently, and colliding
    # with every other model that did the same.
    config_name = _first(cfg, "model.experiment_name", "experiment_name")
    if config_name is not None and str(config_name).strip().lower() in (
            "none", "null", ""):
        config_name = None
    if config_name is None:
        out_dir = _first(cfg, "output_dir", "model.output_dir")
        candidate = Path(str(out_dir)).name if out_dir else None
        # Only believe output_dir when it does not name a container dir.
        if candidate and candidate not in ("models", "results", "."):
            config_name = candidate

    name = fallback_name or config_name
    if not name:
        return None
    canonical = canonical_name(str(name).strip())

    # A config that names a *different* model than the dir it sits in means the run
    # dir was copied or renamed; do not guess which is right.
    if config_name and canonical_name(str(config_name).strip()) != canonical:
        return ModelIdentity(
            canonical=canonical, arch=None, protocol=None, norm=None, eps=None,
            init=None, source=f"config:{src_file}", legacy=True, notes="unparsed",
        )

    arch = str(_first(cfg, "model.model", "model") or "").strip() or None
    # The timm arch for a V1 run is e.g. ``convnext_base_v1``; the *store* arch is
    # the family, matching how the CSVs report it and how the dirs are laid out.
    if arch and arch.endswith("_v1"):
        arch = arch[: -len("_v1")]
    if arch not in (None,) and arch_from_name(canonical) and arch != arch_from_name(canonical):
        # Config and folder name disagree on the arch -- do not guess.
        return ModelIdentity(
            canonical=canonical, arch=None, protocol=None, norm=None, eps=None,
            init=None, source=f"config:{src_file}", legacy=True,
            notes="unparsed",
        )
    if arch is None:
        arch = arch_from_name(canonical)

    eps = _as_float(_first(cfg, "attacks.attack_eps", "attack_eps"))
    norm = _norm_of(_first(cfg, "attacks.attack_norm", "attack_norm"))
    protocol = _protocol_of(cfg, eps, canonical)
    init = _init_from_name(canonical)

    if protocol in ("baseline", "dvd_b_baseline", "v1_clean", "v1_noise"):
        norm, eps = None, None

    mismatch = _name_contradicts(canonical, protocol)
    ok = bool(arch and protocol) and not mismatch
    return ModelIdentity(
        canonical=canonical, arch=arch, protocol=protocol, norm=norm, eps=eps,
        init=init, source=f"config:{src_file}",
        legacy=not ok,
        notes="" if ok else "unparsed",
    )


# Markers that a model *name* carries about its protocol, and the protocols they
# are compatible with. Used only as a cross-check on the config-derived protocol:
# a name that says one thing while the config says another is ambiguous, and the
# user's rule is that ambiguity goes to _legacy/unparsed rather than being guessed.
_NAME_MARKERS: tuple[tuple[str, frozenset[str]], ...] = (
    ("gradnorm", frozenset({"gradnorm"})),
    ("v1_clean", frozenset({"v1_clean"})),
    ("v1_noise", frozenset({"v1_noise"})),
    ("trades", frozenset({"trades"})),
    ("baseline", frozenset({"baseline", "dvd_b_baseline"})),
)


def _name_contradicts(canonical: str, protocol: Optional[str]) -> bool:
    if not protocol:
        return True
    for marker, allowed in _NAME_MARKERS:
        if marker in canonical and protocol not in allowed:
            return True
    return False


def _init_from_name(name: str) -> Optional[str]:
    """``..._init1`` -> ``1``. Returns None for the odd ``initbad7`` style names."""
    for part in reversed(name.split("_")):
        if part.startswith("init"):
            suffix = part[len("init"):]
            return suffix if suffix.isdigit() else None
    return None
