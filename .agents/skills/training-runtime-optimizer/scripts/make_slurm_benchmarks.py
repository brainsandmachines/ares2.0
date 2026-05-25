#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SBATCH_DIR = ROOT / "sbatches_botero" / "timing_skill"
CLUSTER_REPO_ROOT = "/home/ashtomer/projects/ares"
SLURM_TRAIN_DIR = "/groups/golan_neurogroup/bml_group/datasets/imagenet/train/"
SLURM_EVAL_DIR = "/groups/golan_neurogroup/bml_group/datasets/imagenet/val/"

GPU_TARGETS = {
    "rtx6000": {
        "label": "rtx6000",
        "partition": "rtx6000",
        "conda_env": "tomer_advtrain",
        "batch_size_multiplier": 2,
    },
    "rtx_pro_6000": {
        "label": "rtx_pro_6000",
        "partition": "rtx_pro_6000",
        "conda_env": "tomer_advtrain_pro",
        "batch_size_multiplier": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additive Slurm timing sbatches only for locally improved runtime ideas")
    parser.add_argument("--summary-csv", default="", help="Optional benchmark_local.sh summary containing both baseline and candidate rows")
    parser.add_argument("--comparison-csv", default="", help="Optional compare_benchmarks.py output")
    parser.add_argument("--batch-sizes", default="128,112,96")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measured-iters", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()
    if not args.summary_csv and not args.comparison_csv:
        parser.error("one of --summary-csv or --comparison-csv is required")
    return args


def sanitize_job_name(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw)


def load_result(path_str: str) -> dict[str, object]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_winners_from_summary(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    baseline_by_key: dict[tuple[str, int], dict[str, object]] = {}
    candidate_best: dict[tuple[str, str], tuple[float, int]] = {}

    for row in rows:
        if row["status"] != "ok":
            continue
        result = load_result(row["result_json"])
        key = (str(result["protocol"]), int(result["batch_size"]))
        if row["candidate"] == "baseline":
            baseline_by_key[key] = result

    for row in rows:
        if row["candidate"] == "baseline" or row["status"] != "ok":
            continue
        result = load_result(row["result_json"])
        key = (str(result["protocol"]), int(result["batch_size"]))
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        base_ips = float(baseline.get("images_per_sec", 0.0))
        cand_ips = float(result.get("images_per_sec", 0.0))
        if cand_ips <= base_ips:
            continue
        winner_key = (str(result["protocol"]), str(result["candidate"]))
        existing = candidate_best.get(winner_key)
        if existing is None or cand_ips > existing[0]:
            candidate_best[winner_key] = (cand_ips, int(result["batch_size"]))

    winners = []
    for (protocol, candidate), (_cand_ips, batch_size) in sorted(candidate_best.items()):
        winners.append({"protocol": protocol, "candidate": candidate, "batch_size": str(batch_size)})
    return winners


def load_winners_from_comparison(comparison_csv: Path) -> list[dict[str, str]]:
    winners: dict[tuple[str, str], tuple[float, int]] = {}
    with comparison_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("candidate_status") != "ok":
                continue
            speedup = float(row.get("speedup_over_baseline", 0.0))
            if speedup <= 1.0:
                continue
            key = (row["protocol"], row["candidate"])
            batch_size = int(row["batch_size"])
            existing = winners.get(key)
            if existing is None or speedup > existing[0]:
                winners[key] = (speedup, batch_size)

    return [
        {"protocol": protocol, "candidate": candidate, "batch_size": str(batch_size)}
        for (protocol, candidate), (_speedup, batch_size) in sorted(winners.items())
    ]


def load_winners(args: argparse.Namespace) -> list[dict[str, str]]:
    winners: dict[tuple[str, str], dict[str, str]] = {}
    if args.summary_csv:
        for winner in load_winners_from_summary(Path(args.summary_csv)):
            winners[(winner["protocol"], winner["candidate"])] = winner
    if args.comparison_csv:
        for winner in load_winners_from_comparison(Path(args.comparison_csv)):
            winners[(winner["protocol"], winner["candidate"])] = winner
    return list(winners.values())


def scale_batch_sizes(batch_sizes: str, multiplier: int) -> str:
    values = []
    for item in batch_sizes.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(str(int(item) * multiplier))
    return ",".join(values)


def render_sbatch(protocol: str, candidate: str, gpu_key: str, batch_sizes: str, warmup_iters: int, measured_iters: int, num_workers: int) -> str:
    gpu = GPU_TARGETS[gpu_key]
    job_name = sanitize_job_name(f"{protocol}_{candidate}_{gpu['label']}")
    output_root = f"/home/ashtomer/projects/ares/outs/timing_testing/{job_name}"
    slurm_batch_sizes = scale_batch_sizes(batch_sizes, int(gpu["batch_size_multiplier"]))
    next_step = (
        "If this compile candidate wins, run full-epoch compile validation before changing training code."
        if "torch_compile" in candidate
        else "If this candidate wins against the matching Slurm baseline, it is eligible for a protocol-specific code/config change."
    )
    return f"""#!/bin/bash
# Runtime-idea verification sbatch: emitted only after a local win for this protocol.
# This verifies {measured_iters} measured forward/backward training iterations on the target Slurm GPU.
# Batch size is {gpu['batch_size_multiplier']}x the local RTX 4090 batch size used for this protocol.
# Decision rule: {next_step}
#SBATCH --job-name={job_name}
#SBATCH --time=14-00:00:00
#SBATCH --partition={gpu['partition']}
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output=/home/ashtomer/projects/ares/outs/timing_testing/{job_name}%a.out

set -euo pipefail

module load anaconda
source activate {gpu['conda_env']}

REPO_ROOT="{CLUSTER_REPO_ROOT}"
cd "${{REPO_ROOT}}"

TRAIN_DIR="{SLURM_TRAIN_DIR}"
EVAL_DIR="{SLURM_EVAL_DIR}"

mkdir -p "{output_root}"

bash "${{REPO_ROOT}}/.agents/skills/training-runtime-optimizer/scripts/benchmark_local.sh" \\
  --protocol "{protocol}" \\
  --candidate "{candidate}" \\
  --output-dir "{output_root}" \\
  --train-dir "${{TRAIN_DIR}}" \\
  --eval-dir "${{EVAL_DIR}}" \\
  --batch-sizes "{slurm_batch_sizes}" \\
  --warmup-iters "{warmup_iters}" \\
  --measured-iters "{measured_iters}" \\
  --num-workers "{num_workers}"
"""


def main() -> int:
    args = parse_args()
    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    winners = load_winners(args)
    manifest = []

    for winner in winners:
        for gpu_key in GPU_TARGETS:
            content = render_sbatch(
                protocol=winner["protocol"],
                candidate=winner["candidate"],
                gpu_key=gpu_key,
                batch_sizes=winner.get("batch_size", args.batch_sizes),
                warmup_iters=args.warmup_iters,
                measured_iters=args.measured_iters,
                num_workers=args.num_workers,
            )
            job_name = sanitize_job_name(f"{winner['protocol']}_{winner['candidate']}_{gpu_key}")
            path = SBATCH_DIR / f"{job_name}.sbatch"
            path.write_text(content, encoding="utf-8")
            manifest.append(
                {
                    "protocol": winner["protocol"],
                    "candidate": winner["candidate"],
                    "gpu": gpu_key,
                    "sbatch": str(path),
                }
            )

    (SBATCH_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
