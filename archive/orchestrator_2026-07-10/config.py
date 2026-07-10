"""Configuration for the botero orchestrator, loaded from ``orchestrator/.env``.

All cluster-specific, environment-dependent values live here so the rest of the
code stays deterministic and testable. The two genuinely environment-specific
open items are flagged below: the shared-mount DB path and the ssh alias.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .db import NODE_CAPACITY


def _load_dotenv(path: str) -> dict[str, str]:
    """Minimal KEY=VALUE .env parser (no external dependency)."""
    out: dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            out[key.strip()] = val.strip().strip('"').strip("'")
    return out


@dataclass
class Config:
    # --- DB (the shared-mount SQLite; see "open items" in the plan) ---
    # db_path: path as seen from BOTERO (orchestrator reads/writes here).
    db_path: str = "/home/tomer_a/Documents/ares/orchestrator/orchestrator.db"
    # db_path_cluster: path as seen from a CLUSTER job (passed via sbatch --export).
    db_path_cluster: str = "/home/ashtomer/projects/ares/orchestrator/orchestrator.db"

    # --- SSH / cluster ---
    ssh_host: str = "slurm"  # ssh alias for the submit/login node
    cluster_user: str = "ashtomer"
    cluster_repo: str = "/home/ashtomer/projects/ares"

    # --- where job .out/.err logs land, as seen from BOTERO (for failure reads) ---
    logs_root: str = "/home/ashtomer/projects/ares/outs"

    # --- controller array pools ---
    array_size: int = 200          # tasks per pool: --array=1-<size>%<capacity>
    min_remaining: int = 20        # top up when fewer tasks than this remain

    # --- node capacities (default: 6 rtx_pro_6000 + 8 rtx6000; also the %N throttle) ---
    node_capacity: dict[str, int] = field(default_factory=lambda: dict(NODE_CAPACITY))

    # --- failure analyzer toggles ---
    enable_failure_analyzer: bool = True
    failure_lookback_hours: float = 2.0
    recommendations_dir: str = "/home/tomer_a/Documents/ares/orchestrator/recommendations"
    alert_email: Optional[str] = "tomerash98@gmail.com"

    @classmethod
    def load(cls, dotenv_path: Optional[str] = None) -> "Config":
        here = os.path.dirname(os.path.abspath(__file__))
        env = _load_dotenv(dotenv_path or os.path.join(here, ".env"))

        # Expose .env values to os.environ-based readers (notify's SMTP + LLM
        # settings) without overriding anything already set in the real env.
        for _k, _v in env.items():
            os.environ.setdefault(_k, _v)

        def g(key: str, default):
            return env.get(key, default)

        cfg = cls(
            db_path=g("ORCH_DB", cls.db_path),
            db_path_cluster=g("ORCH_DB_CLUSTER", cls.db_path_cluster),
            ssh_host=g("ORCH_SSH_HOST", cls.ssh_host),
            cluster_user=g("ORCH_CLUSTER_USER", cls.cluster_user),
            cluster_repo=g("ORCH_CLUSTER_REPO", cls.cluster_repo),
            logs_root=g("ORCH_LOGS_ROOT", cls.logs_root),
            array_size=int(g("ORCH_ARRAY_SIZE", cls.array_size)),
            min_remaining=int(g("ORCH_MIN_REMAINING", cls.min_remaining)),
            enable_failure_analyzer=str(
                g("ORCH_ENABLE_FAILURE_ANALYZER", "1")
            ).lower() in ("1", "true", "yes"),
            failure_lookback_hours=float(
                g("ORCH_FAILURE_LOOKBACK_HOURS", cls.failure_lookback_hours)
            ),
            recommendations_dir=g("ORCH_RECOMMENDATIONS_DIR", cls.recommendations_dir),
            alert_email=g("ORCH_ALERT_EMAIL", cls.alert_email) or None,
        )
        return cfg
