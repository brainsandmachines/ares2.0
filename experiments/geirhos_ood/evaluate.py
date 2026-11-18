#!/usr/bin/env python3
"""Run the 16 model-vs-human OOD tasks on the HF-zoo checkpoints (local data only).

    python -m experiments.geirhos_ood.evaluate --dry-run [--data-root DIR]
    python -m experiments.geirhos_ood.evaluate --data-root DIR [--tasks ...] [--models ...]
    python -m experiments.geirhos_ood.evaluate --score-only --data-root DIR

Each checkpoint is loaded once (``data_analysis.autoattack_array_eval.load_model``: EMA
weights, ImageNet normalisation inside the model) and runs every ready task with the
shape-bias run's preprocessing (Resize 256 / CenterCrop 224 / ToTensor). Per-trial
decisions go to ``results/trials/<run_dir>__<kind>/<task>.csv`` in upstream's raw-data
column order. A file that already holds every image is skipped unless ``--force``.
Scoring (``metrics.py``) then rebuilds the derived tables from the trial files.

Nothing here downloads: data come only from the given root (see ``data.py``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.geirhos_ood import data as D
from experiments.geirhos_ood.metrics import (TRIAL_COLUMNS, benchmark_summary, human_metrics, model_metrics,
                                             per_condition_accuracy, read_trials)
from experiments.geirhos_ood.models import ModelEntry, load_models, select_models
from experiments.geirhos_ood.tasks import TASKS, TaskSpec, select_tasks

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
REQUIRED_MODULES = ("torch", "torchvision", "PIL", "pandas", "numpy", "matplotlib")

log = logging.getLogger("geirhos_ood")


def trial_path(out_dir: Path, model: ModelEntry, task: TaskSpec) -> Path:
    return out_dir / "trials" / model.model_id / f"{task.task_id}.csv"


def count_rows(path: Path) -> int:
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)


def is_complete(path: Path, n_images: int) -> bool:
    return path.exists() and n_images > 0 and count_rows(path) == n_images


def missing_dependencies() -> list[str]:
    return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXP_DIR, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------- dry run / status

def status_table(models: list[ModelEntry], tasks: list[TaskSpec], root: Path | None,
                 out_dir: Path) -> tuple[pd.DataFrame, dict[str, D.TaskDataStatus], list[str]]:
    """One row per (model, task): ready / done / missing_data / missing_checkpoint / dependency_blocked."""
    data_status = {t.task_id: D.check_task(root, t) for t in tasks}
    deps = missing_dependencies()
    rows = []
    for m in models:
        ckpt_ok = m.checkpoint.is_file()
        for t in tasks:
            ds = data_status[t.task_id]
            if deps:
                status = "dependency_blocked"
            elif not ckpt_ok:
                status = "missing_checkpoint"
            elif not D.is_runnable(ds):
                status = ds.status
            elif is_complete(trial_path(out_dir, m, t), ds.n_images):
                status = "done"
            else:
                status = "ready"
            rows.append({"model_id": m.model_id, "key": m.key, "family": m.family, "task_id": t.task_id,
                         "status": status, "checkpoint": str(m.checkpoint), "data_status": ds.status})
    return pd.DataFrame(rows), data_status, deps


def write_table(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".json").write_text(df.to_json(orient="records", indent=1))


def dry_run(models, tasks, root, out_dir) -> pd.DataFrame:
    table, data_status, deps = status_table(models, tasks, root, out_dir)
    print(f"data root: {root if root else '(none: pass --data-root or set $' + D.ENV_DATA_ROOT + ')'}")
    print(f"missing dependencies: {deps or 'none'}")
    print(f"models: {len(models)} ({sum(m.checkpoint.is_file() for m in models)} checkpoints present)")
    print(f"\n{'task':20s} {'data':16s} {'images':>12s} {'humans':>8s}  model cells")
    for t in tasks:
        ds = data_status[t.task_id]
        cells = table[table.task_id == t.task_id].status.value_counts().to_dict()
        print(f"{t.task_id:20s} {ds.status:16s} {ds.n_images:>5d}/{ds.n_images_expected!s:>6s} "
              f"{ds.n_human_files:>3d}/{ds.n_human_expected!s:>3s}  {cells}"
              + (f"  [{ds.detail}]" if ds.detail else ""))
    write_table(table, out_dir / "dry_run_status")
    write_table(pd.DataFrame([s.as_dict() for s in data_status.values()]), out_dir / "data_status")
    print(f"\nstatus written to {out_dir / 'dry_run_status.csv'}")
    return table


# ---------------------------------------------------------------- inference

def _decide(model, images, device):
    import torch

    from data_analysis.shape_bias.mapping import aggregate_to_16

    with torch.no_grad():
        probs = torch.softmax(model(images.to(device, non_blocking=True)), dim=1)
        return torch.stack([aggregate_to_16(probs, "mean"), aggregate_to_16(probs, "max")], dim=1).cpu()


def evaluate_task(model, model_entry: ModelEntry, task: TaskSpec, root: Path, out_path: Path, device,
                  batch_size: int, min_batch: int, num_workers: int) -> int:
    from torch.utils.data import DataLoader

    from data_analysis.shape_bias.mapping import CATEGORIES
    from experiments.geirhos_ood.gpu import forward_with_backoff

    dataset = D.OODTaskDataset(root, task)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    rows = []
    for images, idx in loader:
        scores = forward_with_backoff(
            lambda x: _decide(model, x, device), images, min_batch,
            on_backoff=lambda n: log.warning("OOM on %s/%s: retrying with batch %d", model_entry.model_id,
                                             task.task_id, n))
        for i, s in zip(idx.tolist(), scores):
            sample = dataset.samples[i]
            rows.append({
                "subj": model_entry.model_id,
                "session": int(sample["session"].split("-")[-1]),
                "trial": i + 1,
                "rt": "NaN",
                "object_response": CATEGORIES[int(s[0].argmax())],
                "category": sample["category"],
                "condition": sample["condition"],
                "imagename": sample["imagename"],
                "object_response_max": CATEGORIES[int(s[1].argmax())],
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=TRIAL_COLUMNS).to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    return len(rows)


def run(models, tasks, root, out_dir, args) -> None:
    import torch

    from data_analysis.autoattack_array_eval import load_model
    from experiments.geirhos_ood.gpu import GPUBusyError, apply_budget, check_free_memory

    table, data_status, deps = status_table(models, tasks, root, out_dir)
    if deps:
        raise SystemExit(f"missing dependencies: {deps}")
    todo = table[table.status == "ready"] if not args.force else table[table.status.isin(["ready", "done"])]
    log.info("%d model x task cells to run (%d models)", len(todo), todo.model_id.nunique())
    if todo.empty:
        return
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_mem_budget_gb:
        frac = apply_budget(args.gpu_mem_budget_gb, device)
        log.info("GPU budget %.1f GB (fraction %.3f)", args.gpu_mem_budget_gb, frac)

    by_id = {m.model_id: m for m in models}
    failures = []
    for i, (model_id, cells) in enumerate(todo.groupby("model_id", sort=False), 1):
        entry = by_id[model_id]
        if device.type == "cuda":
            try:
                free, total = check_free_memory(args.gpu_need_gb, args.gpu_margin_gb)
            except GPUBusyError as err:
                log.error("stopping before %s: %s", model_id, err)
                failures.append({"model_id": model_id, "task_id": "*", "error": str(err)})
                break
        model, eval_cfg, state_key = load_model(entry.checkpoint, device)
        if int(getattr(eval_cfg, "input_size", 224)) != 224:
            log.warning("%s: checkpoint input_size=%s, evaluated at 224 like the shape-bias run",
                        model_id, eval_cfg.input_size)
        for task_id in cells.task_id:
            task = TASKS[task_id]
            try:
                n = evaluate_task(model, entry, task, root, trial_path(out_dir, entry, task), device,
                                  args.batch_size, args.min_batch, args.num_workers)
                log.info("[%d/%d] %s %s: %d trials (%s)", i, todo.model_id.nunique(), model_id, task_id, n,
                         state_key)
            except Exception as err:  # one failed cell must not stop the others
                log.exception("%s %s failed", model_id, task_id)
                failures.append({"model_id": model_id, "task_id": task_id, "error": repr(err)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if failures:
        fail_path = out_dir / "logs" / f"failures_{args.stamp}.json"
        fail_path.write_text(json.dumps(failures, indent=1))
        log.warning("%d failures -> %s", len(failures), fail_path)


# ---------------------------------------------------------------- scoring

def score(models, tasks, root, out_dir) -> pd.DataFrame:
    """Rebuild the derived tables from trial files; unmeasured cells stay NaN with a status."""
    data_status = {t.task_id: D.check_task(root, t) for t in tasks}
    humans = {}
    human_rows, rows, cond_rows = [], [], []
    for t in tasks:
        files = D.human_csvs(root, t) if root else []
        humans[t.task_id] = read_trials(files) if files else None
        if humans[t.task_id] is not None:
            human_rows.append({"task_id": t.task_id, **human_metrics(humans[t.task_id], t)})
            pc = per_condition_accuracy(humans[t.task_id], t)
            cond_rows.append(pc.assign(model_id="humans", task_id=t.task_id))
    for m in models:
        for t in tasks:
            path = trial_path(out_dir, m, t)
            ds = data_status[t.task_id]
            base = {"model_id": m.model_id, "key": m.key, "family": m.family, "generator": m.generator,
                    "arch": m.arch, "checkpoint_kind": m.kind, "checkpoint": str(m.checkpoint),
                    "task_id": t.task_id}
            if not is_complete(path, ds.n_images):
                status = ds.status if not D.is_runnable(ds) else ("partial" if path.exists() else "not_run")
                rows.append({**base, "status": status})
                continue
            df = read_trials(path)
            rows.append({**base, "status": "done", **model_metrics(df, humans[t.task_id], t)})
            for resp, agg in (("object_response", "mean"), ("object_response_max", "max")):
                cond_rows.append(per_condition_accuracy(df, t, resp).assign(model_id=m.model_id, task_id=t.task_id,
                                                                            aggregation=agg))
    task_rows = pd.DataFrame(rows)
    write_table(task_rows, out_dir / "task_metrics")
    if human_rows:
        write_table(pd.DataFrame(human_rows), out_dir / "human_metrics")
    if cond_rows:
        write_table(pd.concat(cond_rows, ignore_index=True), out_dir / "per_condition_accuracy")
    write_table(benchmark_summary(task_rows), out_dir / "benchmark_summary_16tasks")
    done = task_rows.status.eq("done").sum()
    print(f"scored {done}/{len(task_rows)} model x task cells -> {out_dir / 'task_metrics.csv'}")
    return task_rows


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", help=f"model-vs-human root (datasets/, raw-data/); else ${D.ENV_DATA_ROOT}")
    p.add_argument("--tasks", nargs="+", help="subset of the 16 task ids")
    p.add_argument("--models", nargs="+", help="subset by zoo key, run dir or <run_dir>__<kind>")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--dry-run", action="store_true", help="report readiness only; no model is loaded")
    p.add_argument("--score-only", action="store_true", help="rebuild metric tables from existing trial files")
    p.add_argument("--force", action="store_true", help="re-run cells that already have complete trial files")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-batch", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gpu-mem-budget-gb", type=float, default=None,
                   help="cap this process's CUDA allocator (to share the GPU with another job)")
    p.add_argument("--gpu-need-gb", type=float, default=4.0, help="free memory required before each model load")
    p.add_argument("--gpu-margin-gb", type=float, default=3.0, help="headroom left for other GPU jobs")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = D.resolve_data_root(args.data_root)
    tasks = select_tasks(args.tasks)
    models = select_models(load_models(), args.models)
    if args.dry_run:
        dry_run(models, tasks, root, args.out_dir)
        return

    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "configs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out_dir / "logs" / f"run_{args.stamp}.log")])
    config = {
        "time": args.stamp, "git_sha": git_sha(), "argv": sys.argv, "data_root": str(root),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "preprocessing": "Resize(256) -> CenterCrop(224) -> ToTensor (bilinear); normalisation inside the "
                         "model wrapper (data_analysis.autoattack_array_eval.load_model, EMA weights)",
        "decision": "softmax -> mean (bar) / max (dot) probability per 16-class category -> argmax",
        "tasks": [t.task_id for t in tasks],
        "data_status": [D.check_task(root, t).as_dict() for t in tasks],
        "models": [{"model_id": m.model_id, "key": m.key, "checkpoint": str(m.checkpoint),
                    "family": m.family, "source_json": m.source_json} for m in models],
    }
    (args.out_dir / "configs" / f"run_{args.stamp}.json").write_text(json.dumps(config, indent=1))
    if root is None:
        raise SystemExit(f"no data root: pass --data-root or set ${D.ENV_DATA_ROOT}")
    if not args.score_only:
        run(models, tasks, root, args.out_dir, args)
    score(models, tasks, root, args.out_dir)


if __name__ == "__main__":
    main()
