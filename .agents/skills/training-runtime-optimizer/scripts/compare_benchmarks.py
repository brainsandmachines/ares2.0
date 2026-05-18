#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate benchmark artifacts")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def load_results(root: Path) -> list[dict[str, object]]:
    results = []
    for result_json in root.rglob("result.json"):
        results.append(json.loads(result_json.read_text(encoding="utf-8")))
    return results


def key_for(result: dict[str, object]) -> tuple[str, int]:
    return str(result["protocol"]), int(result["batch_size"])


def main() -> int:
    args = parse_args()
    baseline = {key_for(result): result for result in load_results(Path(args.baseline_dir))}
    candidate = {key_for(result): result for result in load_results(Path(args.candidate_dir))}

    rows = []
    for key, cand in sorted(candidate.items()):
        base = baseline.get(key)
        if base is None:
            continue
        base_ips = float(base.get("images_per_sec", 0.0))
        cand_ips = float(cand.get("images_per_sec", 0.0))
        speedup = cand_ips / base_ips if base_ips > 0 else 0.0
        rows.append(
            {
                "protocol": cand["protocol"],
                "batch_size": cand["batch_size"],
                "candidate": cand["candidate"],
                "gpu_name": cand.get("gpu_name", ""),
                "baseline_images_per_sec": f"{base_ips:.4f}",
                "candidate_images_per_sec": f"{cand_ips:.4f}",
                "speedup_over_baseline": f"{speedup:.4f}",
                "baseline_max_memory_bytes": int(base.get("max_memory_bytes", 0)),
                "candidate_max_memory_bytes": int(cand.get("max_memory_bytes", 0)),
                "candidate_final_loss": f"{float(cand.get('final_loss', 0.0)):.6f}",
                "candidate_status": cand.get("status", ""),
            }
        )

    output_csv = Path(args.output_csv) if args.output_csv else Path(args.candidate_dir) / "comparison.csv"
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
