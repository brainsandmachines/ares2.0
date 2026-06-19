"""Import already-trained models from ``results/models`` into ``models_queue``.

Scans a models root (readable from botero via the sshfs mount), inspects each
model directory's on-disk state, and upserts a queue row in the right stage so
the orchestrator resumes where each model actually is instead of retraining:

    not finished training        -> TRAIN       (PENDING, current_epoch set)
    trained, AA missing/partial  -> AA_SWEEP     (PENDING)
    AA done, comparison plot gone -> PLOT_RESULTS (PENDING)
    everything present            -> COMPLETED

Inspection is torch-free (epoch from summary.csv, AA completeness from the sweep
CSVs, checkpoints by file existence), mirroring data_analysis/autoattack_array_eval.py.
It is intentionally conservative: a borderline "AA incomplete" just means the
AA_SWEEP stage re-checks and skips quickly at runtime (skip-if-complete).

Usage (on botero, against the mount):
    python -m orchestrator.import_existing --db <ORCH_DB> \
        --scan-root /home/tomer_a/slurm_mount/projects/ares/results/models \
        --cluster-root /home/ashtomer/projects/ares/results/models --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Optional

from .db import (
    OrchestratorDB,
    STAGE_AA_SWEEP,
    STAGE_COMPLETED,
    STAGE_PLOT_RESULTS,
    STAGE_TRAIN,
)
from . import naming

# Mirror of data_analysis/autoattack_array_eval.py constants (kept torch-free).
NORMS = ("linf", "l2", "l1")
EPS_INPUTS = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)
AA_BASE_CSV = "autoattack_sweep_results.csv"
CKPT_CANDIDATES = {
    "best": ("model_best.pth.tar", "model_best.pth"),
    "last": ("last.pth.tar", "last.pth", "last.pth.model"),
    "advbest": ("model_best_adv.pth.tar",),
}
CKPT_SUFFIX = {"best": "", "last": "last", "advbest": "advbest"}
CONT_RE = re.compile(r"_cont[0-9.]+to[0-9.]+_(direct|ramp)_init[0-9]+$")


def aa_csv_for_kind(kind: str) -> str:
    suffix = CKPT_SUFFIX[kind]
    if not suffix:
        return AA_BASE_CSV
    stem, ext = os.path.splitext(AA_BASE_CSV)
    return f"{stem}_{suffix}{ext}"


def epoch_from_summary(model_dir: Path) -> Optional[int]:
    summary = model_dir / "summary.csv"
    if not summary.exists():
        return None
    last = None
    try:
        with summary.open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                raw = row.get("epoch")
                if raw in (None, ""):
                    continue
                try:
                    last = int(float(raw))
                except Exception:
                    continue
    except Exception:
        return None
    return last


def ckpt_exists(model_dir: Path, kind: str) -> bool:
    return any((model_dir / name).exists() for name in CKPT_CANDIDATES[kind])


def aa_complete(model_dir: Path, kind: str) -> bool:
    csv_path = model_dir / aa_csv_for_kind(kind)
    if not csv_path.exists():
        return False
    expected = {(n, float(e)) for n in NORMS for e in EPS_INPUTS}
    found: set[tuple[str, float]] = set()
    try:
        with csv_path.open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                norm = (row.get("attack_norm") or "").strip()
                raw_eps = row.get("epsilon_input")
                if not norm or raw_eps in (None, ""):
                    continue
                try:
                    found.add((norm, float(raw_eps)))
                except Exception:
                    continue
    except Exception:
        return False
    return expected.issubset(found)


def parse_dirname(name: str):
    """Return (job_name, launcher, warmup, linear, plateau) or None."""
    if name.startswith("convnext_small_"):
        job_name = name[len("convnext_small_"):]
    elif name.startswith("convnext_base_") or name.startswith("convnext_large_"):
        job_name = name  # launchers take the full prefixed name as the job name
    else:
        return None
    m = CONT_RE.search(name)
    if m:
        launcher = "eps_curriculum"
        warmup, linear, plateau = (2, 0, 28) if m.group(1) == "direct" else (3, 17, 15)
    else:
        launcher = "golan-trainmodels"
        warmup, linear, plateau = 0, 0, 150  # standard training length (bump to add more)
    return job_name, launcher, warmup, linear, plateau


def classify(model_dir: Path, total_epochs: int):
    """Return (stage, status, current_epoch)."""
    epoch = epoch_from_summary(model_dir) or 0
    if epoch < total_epochs:
        return STAGE_TRAIN, "PENDING", epoch
    pending_aa = [
        k for k in ("best", "last", "advbest")
        if ckpt_exists(model_dir, k) and not aa_complete(model_dir, k)
    ]
    if pending_aa:
        return STAGE_AA_SWEEP, "PENDING", epoch
    plot = model_dir / f"autoattack_eval_comparation_{model_dir.name}.png"
    if not plot.exists():
        return STAGE_PLOT_RESULTS, "PENDING", epoch
    return STAGE_COMPLETED, "COMPLETED", epoch


def import_models(db_path, scan_root, cluster_root, only=None, dry_run=False):
    scan_root = Path(scan_root)
    db = None if dry_run else OrchestratorDB(db_path)
    results = []
    for entry in sorted(os.listdir(scan_root)):
        model_dir = scan_root / entry
        if not model_dir.is_dir():
            continue
        if only and not any(tok in entry for tok in only):
            continue
        parsed = parse_dirname(entry)
        if parsed is None:
            continue
        job_name, launcher, warmup, linear, plateau = parsed
        total = warmup + linear + plateau
        stage, status, epoch = classify(model_dir, total)
        cluster_dir = os.path.join(cluster_root, entry)
        results.append((entry, launcher, stage, status, epoch, total, cluster_dir))
        if dry_run:
            continue
        db.upsert_model(
            model_id=job_name, launcher=launcher, job_name=job_name,
            model_dir=cluster_dir, warmup_epochs=warmup, linear_epochs=linear,
            plateau_epochs=plateau,
            arch=naming.arch_from_name(job_name),
            protocol=naming.protocol_from_name(job_name),
            eps=naming.eps_from_name(job_name),
        )
        db.update_epoch(job_name, epoch)
        if status == "COMPLETED":
            db.advance_stage(job_name, STAGE_COMPLETED)
        else:
            db.force_stage(job_name, stage, status=status)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Import existing trained models into models_queue")
    ap.add_argument("--db", required=True)
    ap.add_argument("--scan-root", required=True,
                    help="models root readable here (e.g. the sshfs mount path)")
    ap.add_argument("--cluster-root", default="/home/ashtomer/projects/ares/results/models",
                    help="cluster-native models root to store in model_dir")
    ap.add_argument("--only", default=None,
                    help="comma-separated name substrings to include")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    only = [t for t in (args.only or "").split(",") if t] or None

    rows = import_models(args.db, args.scan_root, args.cluster_root, only, args.dry_run)
    tag = "[dry-run] " if args.dry_run else ""
    hdr = f"{'model_dir':40} {'launcher':18} {'stage':13} {'status':9} {'epoch':>11}"
    print(hdr); print("-" * len(hdr))
    for name, launcher, stage, status, epoch, total, _ in rows:
        print(f"{name:40.40} {launcher:18} {stage:13} {status:9} {f'{epoch}/{total}':>11}")
    print(f"\n{tag}{len(rows)} models")


if __name__ == "__main__":
    main()
