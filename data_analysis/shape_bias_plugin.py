"""End-of-training shape-vs-texture bias plug-in.

Runs the Geirhos cue-conflict shape-bias analysis on a finished run's three
checkpoints (``last`` / ``best`` / ``advbest``) and appends one row per checkpoint to
a shared CSV (``data_analysis/shape_bias_results.csv``). Called once, from the tail of
training's ``main()`` after the final eval — see ``run_shape_bias_analysis``.

Design goals: a pure plug-in (no change to how the score is computed) and total
safety (any failure only logs a warning; it must never break the training run). The
scoring code is vendored under ``data_analysis/shape_bias/`` so it ships with ares and
runs on the aircc/slurm clusters.
"""

from __future__ import annotations

import csv
import datetime as dt
import fcntl
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_logger = logging.getLogger(__name__)

# Stimuli live outside git (133 MB), one dir per machine. The matched machine also
# names the output CSV (per-system file) so aircc/slurm/botero never write the same
# file — no git merge conflicts, and results stay separable when aggregated. First
# existing path wins.
STIMULI_CANDIDATES = [
    ("botero", "/home/tomer_a/Documents/shape-bias-analysis/data/style-transfer-preprocessed-512"),
    ("aircc", "/shared/cycle2_bgu_golan_prj/datasets/shape-bias"),
    ("slurm", "/groups/golan_neurogroup/bml_group/datasets/shape-bias"),
]

CHECKPOINT_KINDS = ("last", "best", "advbest")
# Written per-system (e.g. shape_bias_results_aircc.csv) and gitignored — see .gitignore.
CSV_DIR = Path(__file__).resolve().parent
CSV_FIELDS = ["system", "model_name", "checkpoint", "time", "score", "score_max"]


def _append_row(csv_path: Path, row: dict) -> None:
    """Append one result row under an exclusive file lock (concurrency-safe)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: row.get(k, "") for k in CSV_FIELDS}
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            write_header = f.tell() == 0
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(ordered)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _score_checkpoint(model, loader, device, aggregate_to_16, decisions_from_scores, classify_decision):
    """One forward pass over the stimuli; return (rows_mean, rows_max) for summarize."""
    rows_mean, rows_max = [], []
    with torch.no_grad():
        for images, samples in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            dec_mean = decisions_from_scores(aggregate_to_16(probs, "mean"))
            dec_max = decisions_from_scores(aggregate_to_16(probs, "max"))
            for i in range(images.shape[0]):
                shape_label = samples["shape_label"][i]
                texture_label = samples["texture_label"][i]
                outcome_mean, _ = classify_decision(shape_label, texture_label, dec_mean[i])
                outcome_max, _ = classify_decision(shape_label, texture_label, dec_max[i])
                rows_mean.append({"shape_label": shape_label, "texture_label": texture_label, "outcome": outcome_mean})
                rows_max.append({"shape_label": shape_label, "texture_label": texture_label, "outcome": outcome_max})
    return rows_mean, rows_max


def run_shape_bias_analysis(cfg, output_dir):
    """Score the run's checkpoints on the shape-bias stimuli and append to the CSV.

    ``cfg`` is accepted for the training call site but unused. Fully guarded: any
    failure is logged and swallowed so it can never break training.
    """
    try:
        # Vendored scoring code + existing ares checkpoint utilities.
        from data_analysis.shape_bias.dataset import CueConflictDataset
        from data_analysis.shape_bias.mapping import aggregate_to_16, decisions_from_scores
        from data_analysis.shape_bias.metrics import classify_decision, summarize_image_rows
        from data_analysis.autoattack_array_eval import find_checkpoint_for_kind, load_model

        system, stimuli_root = next(((s, p) for s, p in STIMULI_CANDIDATES if os.path.isdir(p)), (None, None))
        if stimuli_root is None:
            _logger.warning("Shape-bias analysis skipped: no stimuli dir found in %s",
                            [p for _, p in STIMULI_CANDIDATES])
            return

        csv_path = CSV_DIR / f"shape_bias_results_{system}.csv"
        model_name = os.path.basename(os.path.normpath(str(output_dir)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = DataLoader(CueConflictDataset(stimuli_root), batch_size=64, shuffle=False, num_workers=4)
        model_dir = Path(output_dir)

        for kind in CHECKPOINT_KINDS:
            ckpt = find_checkpoint_for_kind(model_dir, kind)
            if ckpt is None:
                _logger.info("Shape-bias: no '%s' checkpoint in %s; skipping.", kind, output_dir)
                continue
            try:
                model, _, _ = load_model(ckpt, device)
                rows_mean, rows_max = _score_checkpoint(
                    model, loader, device, aggregate_to_16, decisions_from_scores, classify_decision
                )
                score = summarize_image_rows(rows_mean)["shape_bias"]
                score_max = summarize_image_rows(rows_max)["shape_bias"]
                _append_row(csv_path, {
                    "system": system,
                    "model_name": model_name,
                    "checkpoint": kind,
                    "time": dt.datetime.now().isoformat(timespec="seconds"),
                    "score": score,
                    "score_max": score_max,
                })
                _logger.info("Shape-bias %s/%s: mean=%.4f max=%.4f", model_name, kind, score, score_max)
            except Exception:
                _logger.warning("Shape-bias analysis failed for '%s' checkpoint (%s).", kind, ckpt, exc_info=True)
    except Exception:
        _logger.warning("Shape-bias analysis skipped due to an unexpected error.", exc_info=True)
