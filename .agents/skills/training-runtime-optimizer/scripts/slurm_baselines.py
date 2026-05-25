#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[4]
SBATCH_DIR = ROOT / "sbatches_botero" / "timing_skill"
BASELINE_SBATCH_DIR = SBATCH_DIR / "baselines"
BASELINE_OUTPUT_ROOT = ROOT / "outs" / "timing_testing" / "slurm_baselines"
DB_DIR = BASELINE_OUTPUT_ROOT / "db"
DEFAULT_JSONL = DB_DIR / "slurm_baselines.jsonl"
DEFAULT_CSV = DB_DIR / "slurm_baselines.csv"
DEFAULT_COMPARISON_CSV = ROOT / "outs" / "timing_testing" / "slurm_candidate_comparisons.csv"
DEFAULT_RECOMMENDATION_JSON = ROOT / "outs" / "timing_testing" / "slurm_candidate_recommendations.json"

CLUSTER_REPO_ROOT = "/home/ashtomer/projects/ares"
SLURM_TRAIN_DIR = "/groups/golan_neurogroup/bml_group/datasets/imagenet/train/"
SLURM_EVAL_DIR = "/groups/golan_neurogroup/bml_group/datasets/imagenet/val/"

GPU_TARGETS = {
    "rtx6000": {
        "label": "rtx6000",
        "partition": "rtx6000",
        "conda_env": "tomer_advtrain",
    },
    "rtx_pro_6000": {
        "label": "rtx_pro_6000",
        "partition": "rtx_pro_6000",
        "conda_env": "tomer_advtrain_pro",
    },
}

BASELINE_FIELDS = [
    "setup_key",
    "gpu_key",
    "gpu_name",
    "partition",
    "protocol",
    "batch_size",
    "warmup_iters",
    "measured_iters",
    "num_workers",
    "train_dir",
    "eval_dir",
    "conda_env",
    "status",
    "images_per_sec",
    "avg_iter_seconds",
    "avg_data_seconds",
    "max_memory_bytes",
    "final_loss",
    "result_json",
    "output_dir",
    "sbatch_path",
    "slurm_job_id",
]

COMPARISON_FIELDS = [
    "comparison_status",
    "setup_key",
    "protocol",
    "candidate",
    "gpu_key",
    "gpu_name",
    "batch_size",
    "warmup_iters",
    "measured_iters",
    "num_workers",
    "baseline_images_per_sec",
    "candidate_images_per_sec",
    "speedup_over_baseline",
    "baseline_avg_iter_seconds",
    "candidate_avg_iter_seconds",
    "baseline_max_memory_bytes",
    "candidate_max_memory_bytes",
    "baseline_final_loss",
    "candidate_final_loss",
    "baseline_status",
    "candidate_status",
    "baseline_result_json",
    "candidate_result_json",
]


@dataclass(frozen=True)
class Setup:
    gpu_key: str
    protocol: str
    batch_size: int
    warmup_iters: int
    measured_iters: int
    num_workers: int
    train_dir: str = SLURM_TRAIN_DIR
    eval_dir: str = SLURM_EVAL_DIR

    @property
    def conda_env(self) -> str:
        return str(GPU_TARGETS[self.gpu_key]["conda_env"])

    @property
    def partition(self) -> str:
        return str(GPU_TARGETS[self.gpu_key]["partition"])

    @property
    def job_name(self) -> str:
        return sanitize_job_name(
            f"baseline_{self.protocol}_bsz{self.batch_size}_nw{self.num_workers}_{self.gpu_key}"
        )

    @property
    def output_dir(self) -> Path:
        return (
            BASELINE_OUTPUT_ROOT
            / self.gpu_key
            / self.protocol
            / f"bsz_{self.batch_size}_nw_{self.num_workers}"
        )

    @property
    def sbatch_path(self) -> Path:
        return BASELINE_SBATCH_DIR / f"{self.job_name}.sbatch"


def sanitize_job_name(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and use Slurm baseline timing database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit-sbatches", help="Generate deduplicated baseline sbatches")
    emit.add_argument("--sbatch-dir", type=Path, default=SBATCH_DIR)
    emit.add_argument("--output-dir", type=Path, default=BASELINE_SBATCH_DIR)
    emit.add_argument("--manifest", type=Path, default=BASELINE_SBATCH_DIR / "manifest.json")
    emit.add_argument("--gpu-key", choices=sorted(GPU_TARGETS), action="append")

    ingest = subparsers.add_parser("ingest", help="Ingest finished baseline result.json files")
    ingest.add_argument("--baseline-root", type=Path, default=BASELINE_OUTPUT_ROOT)
    ingest.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ingest.add_argument("--csv", type=Path, default=DEFAULT_CSV)

    compare = subparsers.add_parser("compare", help="Compare candidate result.json files to baseline DB")
    compare.add_argument("--candidate-root", type=Path, default=ROOT / "outs" / "timing_testing")
    compare.add_argument("--baseline-jsonl", type=Path, default=DEFAULT_JSONL)
    compare.add_argument("--output-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    compare.add_argument("--recommendation-json", type=Path, default=DEFAULT_RECOMMENDATION_JSON)

    return parser.parse_args()


def strip_quotes(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def extract_option(text: str, option: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(option)}\s+(.+?)(?:\s*\\)?$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return strip_quotes(match.group(1))


def gpu_key_from_sbatch(path: Path, text: str) -> str | None:
    partition_match = re.search(r"^#SBATCH\s+--partition=(\S+)", text, re.MULTILINE)
    partition = partition_match.group(1) if partition_match else ""
    for gpu_key, meta in GPU_TARGETS.items():
        if partition == meta["partition"] or path.name.endswith(f"_{gpu_key}.sbatch"):
            return gpu_key
    return None


def setup_from_candidate_sbatch(path: Path) -> Setup | None:
    text = path.read_text(encoding="utf-8")
    gpu_key = gpu_key_from_sbatch(path, text)
    if gpu_key is None:
        return None

    protocol = extract_option(text, "--protocol")
    batch_sizes = extract_option(text, "--batch-sizes")
    warmup_iters = extract_option(text, "--warmup-iters")
    measured_iters = extract_option(text, "--measured-iters")
    num_workers = extract_option(text, "--num-workers")
    if not all([protocol, batch_sizes, warmup_iters, measured_iters, num_workers]):
        return None

    # Generated verification sbatches use one batch size. If a future sbatch has
    # a fallback ladder, create one baseline row for each possible exact setup.
    first_batch = batch_sizes.split(",")[0].strip()
    return Setup(
        gpu_key=gpu_key,
        protocol=str(protocol),
        batch_size=int(first_batch),
        warmup_iters=int(str(warmup_iters)),
        measured_iters=int(str(measured_iters)),
        num_workers=int(str(num_workers)),
    )


def discover_setups(sbatch_dir: Path, gpu_keys: set[str] | None = None) -> list[Setup]:
    setups: dict[tuple[object, ...], Setup] = {}
    for path in sorted(sbatch_dir.glob("*.sbatch")):
        if path.parent.name == "baselines":
            continue
        setup = setup_from_candidate_sbatch(path)
        if setup is None:
            continue
        if gpu_keys is not None and setup.gpu_key not in gpu_keys:
            continue
        key = (
            setup.gpu_key,
            setup.protocol,
            setup.batch_size,
            setup.warmup_iters,
            setup.measured_iters,
            setup.num_workers,
        )
        setups[key] = setup
    return [setups[key] for key in sorted(setups)]


def setup_key(setup: Setup) -> str:
    payload = {
        "gpu_key": setup.gpu_key,
        "protocol": setup.protocol,
        "batch_size": setup.batch_size,
        "warmup_iters": setup.warmup_iters,
        "measured_iters": setup.measured_iters,
        "num_workers": setup.num_workers,
        "train_dir": setup.train_dir,
        "eval_dir": setup.eval_dir,
        "conda_env": setup.conda_env,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def render_baseline_sbatch(setup: Setup) -> str:
    output_root = str(setup.output_dir)
    stdout_root = str(BASELINE_OUTPUT_ROOT)
    return f"""#!/bin/bash
# Slurm baseline timing sbatch for candidate comparison.
# This runs the unchanged benchmark setup: same protocol, batch size, workers,
# warmup, measured iterations, dataset roots, GPU partition, and conda env.
#SBATCH --job-name={setup.job_name}
#SBATCH --time=14-00:00:00
#SBATCH --partition={setup.partition}
#SBATCH --qos=golan-neuro
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashtomer@post.bgu.ac.il
#SBATCH --output={stdout_root}/{setup.job_name}-%j.out

set -euo pipefail

module load anaconda
source activate {setup.conda_env}

REPO_ROOT="{CLUSTER_REPO_ROOT}"
cd "${{REPO_ROOT}}"

TRAIN_DIR="{setup.train_dir}"
EVAL_DIR="{setup.eval_dir}"

mkdir -p "{output_root}"

bash "${{REPO_ROOT}}/.agents/skills/training-runtime-optimizer/scripts/benchmark_local.sh" \\
  --protocol "{setup.protocol}" \\
  --candidate "baseline" \\
  --output-dir "{output_root}" \\
  --train-dir "${{TRAIN_DIR}}" \\
  --eval-dir "${{EVAL_DIR}}" \\
  --batch-sizes "{setup.batch_size}" \\
  --warmup-iters "{setup.warmup_iters}" \\
  --measured-iters "{setup.measured_iters}" \\
  --num-workers "{setup.num_workers}"
"""


def emit_sbatches(args: argparse.Namespace) -> int:
    gpu_keys = set(args.gpu_key) if args.gpu_key else None
    setups = discover_setups(args.sbatch_dir, gpu_keys)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    BASELINE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = []
    for setup in setups:
        path = args.output_dir / f"{setup.job_name}.sbatch"
        path.write_text(render_baseline_sbatch(setup), encoding="utf-8")
        manifest.append(
            {
                "setup_key": setup_key(setup),
                "gpu_key": setup.gpu_key,
                "partition": setup.partition,
                "protocol": setup.protocol,
                "batch_size": setup.batch_size,
                "warmup_iters": setup.warmup_iters,
                "measured_iters": setup.measured_iters,
                "num_workers": setup.num_workers,
                "conda_env": setup.conda_env,
                "train_dir": setup.train_dir,
                "eval_dir": setup.eval_dir,
                "output_dir": str(setup.output_dir),
                "sbatch_path": str(path),
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(manifest)} baseline sbatches to {args.output_dir}")
    print(f"Wrote manifest to {args.manifest}")
    return 0


def gpu_key_from_result_path(path: Path) -> str | None:
    parts = path.parts
    if "slurm_baselines" in parts:
        idx = parts.index("slurm_baselines")
        if idx + 1 < len(parts) and parts[idx + 1] in GPU_TARGETS:
            return parts[idx + 1]
    for part in parts:
        if part.endswith("_rtx_pro_6000"):
            return "rtx_pro_6000"
        if part.endswith("_rtx6000"):
            return "rtx6000"
    return None


def parse_slurm_job_id(stdout_path: Path) -> str:
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    match = re.search(r"SLURM_JOBID:\s*(\S+)", text)
    return match.group(1) if match else ""


def find_slurm_job_id(setup: Setup) -> str:
    matches = sorted(BASELINE_OUTPUT_ROOT.glob(f"{setup.job_name}-*.out"))
    for match in reversed(matches):
        job_id = parse_slurm_job_id(match)
        if job_id:
            return job_id
        stem = match.stem
        if "-" in stem:
            return stem.rsplit("-", 1)[-1]
    return ""


def row_for_baseline_result(result_json: Path) -> dict[str, object] | None:
    result = json.loads(result_json.read_text(encoding="utf-8"))
    gpu_key = gpu_key_from_result_path(result_json)
    if gpu_key is None:
        return None
    setup = Setup(
        gpu_key=gpu_key,
        protocol=str(result["protocol"]),
        batch_size=int(result["batch_size"]),
        warmup_iters=int(result["warmup_iters"]),
        measured_iters=int(result["measured_iters"]),
        num_workers=int(result["num_workers"]),
    )
    return {
        "setup_key": setup_key(setup),
        "gpu_key": setup.gpu_key,
        "gpu_name": result.get("gpu_name", ""),
        "partition": setup.partition,
        "protocol": setup.protocol,
        "batch_size": setup.batch_size,
        "warmup_iters": setup.warmup_iters,
        "measured_iters": setup.measured_iters,
        "num_workers": setup.num_workers,
        "train_dir": setup.train_dir,
        "eval_dir": setup.eval_dir,
        "conda_env": setup.conda_env,
        "status": result.get("status", ""),
        "images_per_sec": result.get("images_per_sec", ""),
        "avg_iter_seconds": result.get("avg_iter_seconds", ""),
        "avg_data_seconds": result.get("avg_data_seconds", ""),
        "max_memory_bytes": result.get("max_memory_bytes", ""),
        "final_loss": result.get("final_loss", ""),
        "result_json": str(result_json),
        "output_dir": str(result_json.parent),
        "sbatch_path": str(setup.sbatch_path),
        "slurm_job_id": find_slurm_job_id(setup),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ingest(args: argparse.Namespace) -> int:
    rows = []
    for result_json in sorted(args.baseline_root.rglob("result.json")):
        row = row_for_baseline_result(result_json)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda row: (str(row["gpu_key"]), str(row["protocol"]), int(row["batch_size"]), int(row["num_workers"])))
    write_jsonl(args.jsonl, rows)
    write_csv(args.csv, rows, BASELINE_FIELDS)
    print(f"Wrote {len(rows)} baseline rows to {args.jsonl} and {args.csv}")
    return 0


def load_baseline_db(path: Path) -> dict[str, dict[str, object]]:
    baselines = {}
    if not path.exists():
        return baselines
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            baselines[str(row["setup_key"])] = row
    return baselines


def is_candidate_result(path: Path, candidate_root: Path) -> bool:
    if path.name != "result.json":
        return False
    if "slurm_baselines" in path.parts:
        return False
    try:
        relative = path.relative_to(candidate_root)
    except ValueError:
        return False
    return len(relative.parts) >= 4


def comparison_row(result_json: Path, baselines: dict[str, dict[str, object]]) -> dict[str, object] | None:
    result = json.loads(result_json.read_text(encoding="utf-8"))
    if result.get("candidate") == "baseline":
        return None
    gpu_key = gpu_key_from_result_path(result_json)
    if gpu_key is None:
        return None
    setup = Setup(
        gpu_key=gpu_key,
        protocol=str(result["protocol"]),
        batch_size=int(result["batch_size"]),
        warmup_iters=int(result["warmup_iters"]),
        measured_iters=int(result["measured_iters"]),
        num_workers=int(result["num_workers"]),
    )
    key = setup_key(setup)
    baseline = baselines.get(key)
    row = {
        "comparison_status": "ok" if baseline else "missing_baseline",
        "setup_key": key,
        "protocol": result.get("protocol", ""),
        "candidate": result.get("candidate", ""),
        "gpu_key": gpu_key,
        "gpu_name": result.get("gpu_name", ""),
        "batch_size": result.get("batch_size", ""),
        "warmup_iters": result.get("warmup_iters", ""),
        "measured_iters": result.get("measured_iters", ""),
        "num_workers": result.get("num_workers", ""),
        "baseline_images_per_sec": "",
        "candidate_images_per_sec": result.get("images_per_sec", ""),
        "speedup_over_baseline": "",
        "baseline_avg_iter_seconds": "",
        "candidate_avg_iter_seconds": result.get("avg_iter_seconds", ""),
        "baseline_max_memory_bytes": "",
        "candidate_max_memory_bytes": result.get("max_memory_bytes", ""),
        "baseline_final_loss": "",
        "candidate_final_loss": result.get("final_loss", ""),
        "baseline_status": "",
        "candidate_status": result.get("status", ""),
        "baseline_result_json": "",
        "candidate_result_json": str(result_json),
    }
    if baseline:
        base_ips = float(baseline.get("images_per_sec") or 0.0)
        cand_ips = float(result.get("images_per_sec") or 0.0)
        row.update(
            {
                "baseline_images_per_sec": baseline.get("images_per_sec", ""),
                "speedup_over_baseline": f"{cand_ips / base_ips:.6f}" if base_ips > 0 else "0.000000",
                "baseline_avg_iter_seconds": baseline.get("avg_iter_seconds", ""),
                "baseline_max_memory_bytes": baseline.get("max_memory_bytes", ""),
                "baseline_final_loss": baseline.get("final_loss", ""),
                "baseline_status": baseline.get("status", ""),
                "baseline_result_json": baseline.get("result_json", ""),
            }
        )
    return row


def candidate_needs_full_epoch(candidate: str) -> bool:
    return "torch_compile" in candidate


def recommendation_for_row(row: dict[str, object]) -> dict[str, object]:
    evidence_paths = [
        str(row.get("baseline_result_json", "")),
        str(row.get("candidate_result_json", "")),
    ]
    recommendation = "reject_candidate"
    required_next_validation = ""
    proposed_code_change = ""

    if row.get("comparison_status") != "ok":
        reason = "missing matching Slurm baseline; no code change can be recommended from this result"
        required_next_validation = "run or ingest the matching Slurm baseline, then compare again"
    elif row.get("candidate_status") != "ok":
        reason = "candidate Slurm run failed"
    else:
        speedup = float(row.get("speedup_over_baseline") or 0.0)
        candidate = str(row.get("candidate", ""))
        protocol = str(row.get("protocol", ""))
        if speedup <= 1.0:
            reason = "candidate did not beat the Slurm baseline"
        elif candidate_needs_full_epoch(candidate):
            recommendation = "run_full_epoch_validation"
            reason = "compile candidate beat the short Slurm baseline but needs full-epoch compile validation"
            required_next_validation = (
                "run full-epoch compile-vs-no-compile validation and verify compile_applied=true"
            )
        else:
            recommendation = "recommend_code_change"
            reason = "candidate beat the matching Slurm baseline"
            proposed_code_change = f"enable candidate '{candidate}' for protocol '{protocol}'"

    return {
        "stage": "slurm_verification",
        "recommendation": recommendation,
        "reason": reason,
        "required_next_validation": required_next_validation,
        "proposed_code_change": proposed_code_change,
        "evidence_paths": [path for path in evidence_paths if path],
        "protocol": row.get("protocol", ""),
        "candidate": row.get("candidate", ""),
        "gpu_key": row.get("gpu_key", ""),
        "speedup_over_baseline": row.get("speedup_over_baseline", ""),
    }


def compare(args: argparse.Namespace) -> int:
    baselines = load_baseline_db(args.baseline_jsonl)
    rows = []
    for result_json in sorted(args.candidate_root.rglob("result.json")):
        if not is_candidate_result(result_json, args.candidate_root):
            continue
        row = comparison_row(result_json, baselines)
        if row is not None:
            rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row["comparison_status"]),
            str(row["gpu_key"]),
            str(row["protocol"]),
            str(row["candidate"]),
            int(row["batch_size"]),
            int(row["num_workers"]),
        )
    )
    write_csv(args.output_csv, rows, COMPARISON_FIELDS)
    recommendations = [recommendation_for_row(row) for row in rows]
    args.recommendation_json.parent.mkdir(parents=True, exist_ok=True)
    args.recommendation_json.write_text(
        json.dumps(recommendations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ok_count = sum(1 for row in rows if row["comparison_status"] == "ok")
    missing_count = sum(1 for row in rows if row["comparison_status"] == "missing_baseline")
    print(f"Wrote {len(rows)} comparison rows to {args.output_csv} ({ok_count} ok, {missing_count} missing_baseline)")
    print(f"Wrote {len(recommendations)} recommendation rows to {args.recommendation_json}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "emit-sbatches":
        return emit_sbatches(args)
    if args.command == "ingest":
        return ingest(args)
    if args.command == "compare":
        return compare(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
