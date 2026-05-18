#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = ROOT / ".agents" / "skills" / "training-runtime-optimizer"
BENCHMARK_SCRIPT = SKILL_DIR / "scripts" / "benchmark_local.sh"

PROTOCOLS = [
    "pixel_linf_madry",
    "pixel_l2_madry",
    "pixel_l1_madry",
    "pixel_linf_trades_rs_on",
    "pixel_linf_trades_rs_off",
    "pixel_l2_trades_rs_on",
    "pixel_l2_trades_rs_off",
    "pixel_l1_trades_rs_on",
    "pixel_l1_trades_rs_off",
    "gradnorm_l2",
    "v1_l2_madry",
    "v1_l2_trades",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training-runtime-optimizer protocol matrix")
    parser.add_argument(
        "--output-root",
        default=str(SKILL_DIR / "outputs" / "protocol_matrix"),
        help="Directory for raw benchmark outputs and aggregate summaries",
    )
    parser.add_argument(
        "--protocols",
        default=",".join(PROTOCOLS),
        help="CSV list of protocols to benchmark",
    )
    parser.add_argument(
        "--candidates",
        default="baseline,channels_last",
        help="CSV list of candidates to benchmark",
    )
    parser.add_argument(
        "--workers",
        default="6,8",
        help="CSV list of num_workers values to benchmark",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_best_success(summary_csv: Path) -> dict[str, object] | None:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    successes = []
    for row in rows:
        if row["status"] != "ok":
            continue
        result = load_result(Path(row["result_json"]))
        successes.append(result)

    if not successes:
        return None
    return max(successes, key=lambda item: float(item.get("images_per_sec", 0.0)))


def run_benchmark(protocol: str, candidate: str, num_workers: int, output_dir: Path) -> dict[str, object]:
    cmd = [
        "bash",
        str(BENCHMARK_SCRIPT),
        "--protocol",
        protocol,
        "--candidate",
        candidate,
        "--output-dir",
        str(output_dir),
        "--num-workers",
        str(num_workers),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    summary_csv = output_dir / "summary.csv"
    best_result = load_best_success(summary_csv) if summary_csv.exists() else None

    payload: dict[str, object] = {
        "protocol": protocol,
        "candidate": candidate,
        "num_workers": num_workers,
        "command": cmd,
        "returncode": completed.returncode,
        "summary_csv": str(summary_csv),
        "status": "failed",
    }
    if best_result is not None:
        payload.update(
            {
                "status": "ok",
                "batch_size": int(best_result["batch_size"]),
                "images_per_sec": float(best_result["images_per_sec"]),
                "avg_iter_seconds": float(best_result["avg_iter_seconds"]),
                "avg_data_seconds": float(best_result["avg_data_seconds"]),
                "max_memory_bytes": int(best_result["max_memory_bytes"]),
                "result_json": str(
                    output_dir
                    / protocol
                    / candidate
                    / f"bsz_{best_result['batch_size']}"
                    / "result.json"
                ),
            }
        )
    return payload


def summarize_best(results: list[dict[str, object]]) -> list[dict[str, object]]:
    best_rows = []
    for protocol in sorted({str(item["protocol"]) for item in results}):
        candidates = [item for item in results if item["protocol"] == protocol and item["status"] == "ok"]
        if not candidates:
            best_rows.append({"protocol": protocol, "status": "no_success"})
            continue
        winner = max(candidates, key=lambda item: float(item["images_per_sec"]))
        best_rows.append(
            {
                "protocol": protocol,
                "best_candidate": winner["candidate"],
                "best_num_workers": winner["num_workers"],
                "best_batch_size": winner["batch_size"],
                "best_images_per_sec": winner["images_per_sec"],
                "best_avg_iter_seconds": winner["avg_iter_seconds"],
                "best_avg_data_seconds": winner["avg_data_seconds"],
                "best_max_memory_bytes": winner["max_memory_bytes"],
                "result_json": winner["result_json"],
            }
        )
    return best_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    protocols = [item for item in args.protocols.split(",") if item]
    candidates = [item for item in args.candidates.split(",") if item]
    workers = [int(item) for item in args.workers.split(",") if item]

    all_results: list[dict[str, object]] = []
    for protocol in protocols:
        for candidate in candidates:
            for num_workers in workers:
                run_dir = output_root / "runs" / protocol / candidate / f"nw_{num_workers}"
                run_dir.mkdir(parents=True, exist_ok=True)
                result = run_benchmark(protocol, candidate, num_workers, run_dir)
                all_results.append(result)

    matrix_json = output_root / "matrix_results.json"
    matrix_json.write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_root / "matrix_results.csv", all_results)

    best_rows = summarize_best(all_results)
    best_json = output_root / "best_by_protocol.json"
    best_json.write_text(json.dumps(best_rows, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_root / "best_by_protocol.csv", best_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
