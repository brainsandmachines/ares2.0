#!/usr/bin/env python3
"""Score shape bias for the exact experiment checkpoint of each selected model.

The end-of-training plug-in (``shape_bias_plugin.py``) scores every checkpoint kind of a
run, but its AIRCC results were never archived and its slurm rows name vit/swin runs
ambiguously. This re-scores the selected set (``selected_models.py``) on Botero with the
same scoring code, appending to the plug-in's per-system CSV
(``shape_bias_results_botero.csv``). ``model_name`` there is the run dir name.
Resumable: (run dir, kind) pairs already in the CSV are skipped.

python -m data_analysis.shape_bias_selected [--only <run_dir> ...]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os

import torch
from torch.utils.data import DataLoader

from data_analysis.autoattack_array_eval import load_model
from data_analysis.selected_models import resolve
from data_analysis.shape_bias.dataset import CueConflictDataset
from data_analysis.shape_bias.mapping import aggregate_to_16, decisions_from_scores
from data_analysis.shape_bias.metrics import classify_decision, summarize_image_rows
from data_analysis.shape_bias_plugin import CSV_DIR, STIMULI_CANDIDATES, _append_row, _score_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", help="Score only these run dirs (for smoke tests).")
    args = parser.parse_args()

    system, stimuli_root = next((s, p) for s, p in STIMULI_CANDIDATES if os.path.isdir(p))
    csv_path = CSV_DIR / f"shape_bias_results_{system}.csv"
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            done = {(r["model_name"], r["checkpoint"]) for r in csv.DictReader(f)}

    families, baselines = resolve()
    todo = {s.checkpoint: s for s in [*baselines.values(), *(s for ss in families.values() for s in ss)]}
    todo = [s for s in todo.values()
            if (s.run_dir, s.kind) not in done and (not args.only or s.run_dir in args.only)]
    print(f"{len(todo)} checkpoints to score -> {csv_path}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(CueConflictDataset(stimuli_root), batch_size=64, shuffle=False, num_workers=4)
    for i, sel in enumerate(todo, 1):
        model, _, _ = load_model(sel.checkpoint, device)
        rows_mean, rows_max = _score_checkpoint(
            model, loader, device, aggregate_to_16, decisions_from_scores, classify_decision
        )
        score = summarize_image_rows(rows_mean)["shape_bias"]
        score_max = summarize_image_rows(rows_max)["shape_bias"]
        _append_row(csv_path, {
            "system": system,
            "model_name": sel.run_dir,
            "checkpoint": sel.kind,
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "score": score,
            "score_max": score_max,
        })
        print(f"[{i}/{len(todo)}] {sel.run_dir}/{sel.kind}: mean={score:.4f} max={score_max:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
