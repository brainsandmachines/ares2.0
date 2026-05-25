#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = ROOT / ".agents" / "skills" / "training-runtime-optimizer"
STATE_FILE = SKILL_DIR / "state" / "ideas_tested.json"
LOCK_FILE = SKILL_DIR / "state" / "run.lock"
OUTPUT_ROOT = SKILL_DIR / "outputs" / "runs"
BENCHMARK_SCRIPT = SKILL_DIR / "scripts" / "benchmark_local.sh"
CONTRACT_SCRIPT = SKILL_DIR / "scripts" / "run_contract_tests.sh"
CANDIDATE_CONTRACT_SCRIPT = SKILL_DIR / "scripts" / "run_candidate_contract_tests.sh"
COMPARE_SCRIPT = SKILL_DIR / "scripts" / "compare_benchmarks.py"
SLURM_SCRIPT = SKILL_DIR / "scripts" / "make_slurm_benchmarks.py"
IDEA_STATE_SCRIPT = SKILL_DIR / "scripts" / "idea_state.py"

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

GLOBAL_RUNTIME_CANDIDATES = {
    "channels_last",
    "dataloader_tuned",
    "torch_compile",
    "zero_grad_set_to_none",
}

RECOMMENDATIONS = {
    "recommend_code_change",
    "run_full_epoch_validation",
    "reject_candidate",
}


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one scheduled-safe runtime idea cycle. This script does not choose ideas; "
            "the agent supplies one open-ended idea id and candidate command."
        )
    )
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--source", default="repo")
    parser.add_argument("--rationale", default="")
    parser.add_argument("--touched-behavior", default="")
    parser.add_argument("--source-notes", default="")
    parser.add_argument("--knobs-json", default="{}")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--baseline-command", default="")
    parser.add_argument("--candidate-command", default="")
    parser.add_argument(
        "--related-protocols",
        default="auto",
        help=(
            "Comma-separated related protocols to test if the primary protocol wins. "
            "Use 'auto' for built-in global runtime candidates, or 'none' to disable."
        ),
    )
    parser.add_argument("--skip-contract-tests", action="store_true")
    parser.add_argument("--no-slurm", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], decision: dict[str, Any], label: str) -> int:
    decision.setdefault("commands", []).append({"label": label, "command": cmd})
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    decision.setdefault("returncodes", {})[label] = completed.returncode
    return completed.returncode


def render_command(template: str, values: dict[str, str]) -> list[str]:
    rendered = template.format(**values)
    return shlex.split(rendered)


def related_protocols(primary_protocol: str, candidate_name: str, value: str) -> list[str]:
    if value in {"", "none", "false", "off"}:
        return []
    if value == "auto":
        if candidate_name in GLOBAL_RUNTIME_CANDIDATES:
            return [protocol for protocol in PROTOCOLS if protocol != primary_protocol]
        return []
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(PROTOCOLS))
    if unknown:
        raise SystemExit(f"Unknown related protocol(s): {', '.join(unknown)}")
    return [protocol for protocol in requested if protocol != primary_protocol]


def protocol_idea_id(base_idea_id: str, protocol: str) -> str:
    if "__" not in base_idea_id:
        return base_idea_id
    return f"{protocol}__{base_idea_id.split('__', 1)[1]}"


def load_best_result(summary_csv: Path) -> dict[str, Any] | None:
    if not summary_csv.exists():
        return None
    import csv

    successes: list[dict[str, Any]] = []
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            result_path = Path(row["result_json"])
            if result_path.exists():
                successes.append(json.loads(result_path.read_text(encoding="utf-8")))
    if not successes:
        return None
    return max(successes, key=lambda item: float(item.get("images_per_sec", 0.0)))


def combine_comparisons(paths: list[Path], output_csv: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None and reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows.extend(dict(row) for row in reader)
    if not rows or fieldnames is None:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_state(
    decision: dict[str, Any],
    label: str,
    idea_id: str,
    protocol: str,
    knobs_json: str,
    status: str,
    artifacts: dict[str, str],
    reason: str = "",
) -> int:
    return run_cmd(
        [
            sys.executable,
            str(IDEA_STATE_SCRIPT),
            "record-result",
            "--idea-id",
            idea_id,
            "--protocol",
            protocol,
            "--knobs-json",
            knobs_json,
            "--status",
            status,
            "--artifacts-json",
            json.dumps(artifacts, sort_keys=True),
            "--rejection-reason",
            reason,
        ],
        decision,
        label,
    )


def candidate_contract_cmd(protocol: str, candidate: str) -> list[str]:
    return [
        "bash",
        str(CANDIDATE_CONTRACT_SCRIPT),
        "--protocol",
        protocol,
        "--candidate",
        candidate,
    ]


def candidate_needs_full_epoch(candidate: str) -> bool:
    return "torch_compile" in candidate


def make_recommendation(
    *,
    stage: str,
    status: str,
    candidate: str,
    reason: str,
    evidence_paths: list[str],
    speedup: float | None = None,
) -> dict[str, Any]:
    if status != "won":
        recommendation = "reject_candidate"
        required_next_validation = ""
        proposed_code_change = ""
        decision_reason = reason or "candidate did not produce enough valid evidence for a code change"
    elif candidate_needs_full_epoch(candidate):
        recommendation = "run_full_epoch_validation"
        required_next_validation = (
            "run Slurm verification first, then full-epoch compile-vs-no-compile validation if the Slurm result wins"
        )
        proposed_code_change = ""
        decision_reason = "compile candidates need full-epoch evidence before changing training code"
    else:
        recommendation = "recommend_code_change"
        required_next_validation = (
            "run generated Slurm verification sbatches and compare against the matching Slurm baseline"
            if stage == "local_botero"
            else ""
        )
        proposed_code_change = (
            f"after required validation, enable candidate '{candidate}' for the protocol(s) where measured speedup is above baseline"
        )
        decision_reason = reason or "candidate beat the fresh baseline and passed validation"

    payload: dict[str, Any] = {
        "stage": stage,
        "recommendation": recommendation,
        "reason": decision_reason,
        "required_next_validation": required_next_validation,
        "proposed_code_change": proposed_code_change,
        "evidence_paths": evidence_paths,
    }
    if speedup is not None:
        payload["speedup_over_baseline"] = speedup
    assert recommendation in RECOMMENDATIONS
    return payload


def acquire_lock() -> bool:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_at": now_stamp()}) + "\n")
    return True


def release_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    args = parse_args()
    run_dir = Path(args.output_root) / now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_path = run_dir / "decision.json"
    decision: dict[str, Any] = {
        "idea_id": args.idea_id,
        "protocol": args.protocol,
        "candidate_name": args.candidate_name,
        "source": args.source,
        "rationale": args.rationale,
        "touched_behavior": args.touched_behavior,
        "source_notes": args.source_notes,
        "status": "started",
        "run_dir": str(run_dir),
    }

    if not acquire_lock():
        decision.update({"status": "skipped", "reason": "run already active"})
        decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run_cmd(
            [
                sys.executable,
                str(IDEA_STATE_SCRIPT),
                "mark-skipped",
                "--idea-id",
                args.idea_id,
                "--protocol",
                args.protocol,
                "--knobs-json",
                args.knobs_json,
                "--reason",
                "run already active",
            ],
            decision,
            "mark_skipped",
        )
        return 0

    try:
        has_tested = subprocess.run(
            [
                sys.executable,
                str(IDEA_STATE_SCRIPT),
                "has-tested",
                "--idea-id",
                args.idea_id,
                "--protocol",
                args.protocol,
                "--knobs-json",
                args.knobs_json,
            ],
            cwd=ROOT,
            check=False,
        )
        if has_tested.returncode == 0:
            decision.update({"status": "skipped", "reason": "idea already tested"})
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0

        run_cmd(
            [
                sys.executable,
                str(IDEA_STATE_SCRIPT),
                "record-started",
                "--idea-id",
                args.idea_id,
                "--protocol",
                args.protocol,
                "--knobs-json",
                args.knobs_json,
                "--source",
                args.source,
                "--rationale",
                args.rationale,
                "--touched-behavior",
                args.touched_behavior,
                "--source-notes",
                args.source_notes,
            ],
            decision,
            "record_started",
        )

        values = {
            "protocol": args.protocol,
            "candidate": args.candidate_name,
            "run_dir": str(run_dir),
            "baseline_dir": str(run_dir / "baseline"),
            "candidate_dir": str(run_dir / "candidate"),
        }
        baseline_cmd = (
            render_command(args.baseline_command, values)
            if args.baseline_command
            else [
                "bash",
                str(BENCHMARK_SCRIPT),
                "--protocol",
                args.protocol,
                "--candidate",
                "baseline",
                "--output-dir",
                str(run_dir / "baseline"),
            ]
        )
        candidate_cmd = (
            render_command(args.candidate_command, values)
            if args.candidate_command
            else [
                "bash",
                str(BENCHMARK_SCRIPT),
                "--protocol",
                args.protocol,
                "--candidate",
                args.candidate_name,
                "--output-dir",
                str(run_dir / "candidate"),
            ]
        )

        if run_cmd(baseline_cmd, decision, "baseline") != 0:
            decision.update({"status": "failed", "reason": "baseline command failed"})
        elif not args.skip_contract_tests and run_cmd(["bash", str(CONTRACT_SCRIPT)], decision, "contract_before_candidate") != 0:
            decision.update({"status": "failed", "reason": "contract tests failed before candidate"})
        elif run_cmd(candidate_cmd, decision, "candidate") != 0:
            decision.update({"status": "failed", "reason": "candidate command failed"})
        elif not args.skip_contract_tests and run_cmd(["bash", str(CONTRACT_SCRIPT)], decision, "contract_after_candidate") != 0:
            decision.update({"status": "failed", "reason": "contract tests failed after candidate"})
        else:
            comparison_csv = run_dir / "comparison.csv"
            run_cmd(
                [
                    sys.executable,
                    str(COMPARE_SCRIPT),
                    "--baseline-dir",
                    str(run_dir / "baseline"),
                    "--candidate-dir",
                    str(run_dir / "candidate"),
                    "--output-csv",
                    str(comparison_csv),
                ],
                decision,
                "compare",
            )
            baseline_result = load_best_result(run_dir / "baseline" / "summary.csv")
            candidate_result = load_best_result(run_dir / "candidate" / "summary.csv")
            decision["baseline_result"] = baseline_result
            decision["candidate_result"] = candidate_result
            if baseline_result and candidate_result:
                base_ips = float(baseline_result.get("images_per_sec", 0.0))
                cand_ips = float(candidate_result.get("images_per_sec", 0.0))
                speedup = cand_ips / base_ips if base_ips > 0 else 0.0
                decision["speedup_over_baseline"] = speedup
                comparison_paths = [comparison_csv]
                if speedup > 1.0:
                    if not args.skip_contract_tests and run_cmd(
                        candidate_contract_cmd(args.protocol, args.candidate_name),
                        decision,
                        "candidate_contract_primary",
                    ) != 0:
                        decision.update({"status": "failed", "reason": "candidate contract tests failed"})
                    else:
                        decision["status"] = "won"
                    related = related_protocols(args.protocol, args.candidate_name, args.related_protocols)
                    decision["related_protocols"] = related
                    related_results: list[dict[str, Any]] = []
                    for related_protocol in related if decision.get("status") == "won" else []:
                        related_idea_id = protocol_idea_id(args.idea_id, related_protocol)
                        related_run_dir = run_dir / "related" / related_protocol
                        related_baseline_dir = related_run_dir / "baseline"
                        related_candidate_dir = related_run_dir / "candidate"
                        related_comparison_csv = related_run_dir / "comparison.csv"
                        related_artifacts = {
                            "run_dir": str(related_run_dir),
                            "baseline_dir": str(related_baseline_dir),
                            "candidate_dir": str(related_candidate_dir),
                            "comparison_csv": str(related_comparison_csv),
                            "parent_decision_json": str(decision_path),
                        }

                        has_related = subprocess.run(
                            [
                                sys.executable,
                                str(IDEA_STATE_SCRIPT),
                                "has-tested",
                                "--idea-id",
                                related_idea_id,
                                "--protocol",
                                related_protocol,
                                "--knobs-json",
                                args.knobs_json,
                            ],
                            cwd=ROOT,
                            check=False,
                        )
                        if has_related.returncode == 0:
                            related_results.append(
                                {
                                    "protocol": related_protocol,
                                    "idea_id": related_idea_id,
                                    "status": "skipped",
                                    "reason": "idea already tested",
                                }
                            )
                            continue

                        run_cmd(
                            [
                                sys.executable,
                                str(IDEA_STATE_SCRIPT),
                                "record-started",
                                "--idea-id",
                                related_idea_id,
                                "--protocol",
                                related_protocol,
                                "--knobs-json",
                                args.knobs_json,
                                "--source",
                                args.source,
                                "--rationale",
                                args.rationale,
                                "--touched-behavior",
                                args.touched_behavior,
                                "--source-notes",
                                f"{args.source_notes} Related-protocol sweep from {args.protocol}.",
                            ],
                            decision,
                            f"record_started_related_{related_protocol}",
                        )

                        related_values = {
                            "protocol": related_protocol,
                            "candidate": args.candidate_name,
                            "run_dir": str(related_run_dir),
                            "baseline_dir": str(related_baseline_dir),
                            "candidate_dir": str(related_candidate_dir),
                        }
                        related_baseline_cmd = (
                            render_command(args.baseline_command, related_values)
                            if args.baseline_command
                            else [
                                "bash",
                                str(BENCHMARK_SCRIPT),
                                "--protocol",
                                related_protocol,
                                "--candidate",
                                "baseline",
                                "--output-dir",
                                str(related_baseline_dir),
                            ]
                        )
                        related_candidate_cmd = (
                            render_command(args.candidate_command, related_values)
                            if args.candidate_command
                            else [
                                "bash",
                                str(BENCHMARK_SCRIPT),
                                "--protocol",
                                related_protocol,
                                "--candidate",
                                args.candidate_name,
                                "--output-dir",
                                str(related_candidate_dir),
                            ]
                        )

                        related_status = "failed"
                        related_reason = ""
                        related_result_appended = False
                        if run_cmd(related_baseline_cmd, decision, f"baseline_related_{related_protocol}") != 0:
                            related_reason = "related baseline command failed"
                        elif run_cmd(related_candidate_cmd, decision, f"candidate_related_{related_protocol}") != 0:
                            related_reason = "related candidate command failed"
                        else:
                            run_cmd(
                                [
                                    sys.executable,
                                    str(COMPARE_SCRIPT),
                                    "--baseline-dir",
                                    str(related_baseline_dir),
                                    "--candidate-dir",
                                    str(related_candidate_dir),
                                    "--output-csv",
                                    str(related_comparison_csv),
                                ],
                                decision,
                                f"compare_related_{related_protocol}",
                            )
                            related_baseline_result = load_best_result(related_baseline_dir / "summary.csv")
                            related_candidate_result = load_best_result(related_candidate_dir / "summary.csv")
                            if related_baseline_result and related_candidate_result:
                                related_base_ips = float(related_baseline_result.get("images_per_sec", 0.0))
                                related_cand_ips = float(related_candidate_result.get("images_per_sec", 0.0))
                                related_speedup = (
                                    related_cand_ips / related_base_ips if related_base_ips > 0 else 0.0
                                )
                                related_status = "won" if related_speedup > 1.0 else "neutral"
                                if related_status == "neutral":
                                    related_reason = "candidate did not beat related baseline"
                                elif not args.skip_contract_tests and run_cmd(
                                    candidate_contract_cmd(related_protocol, args.candidate_name),
                                    decision,
                                    f"candidate_contract_related_{related_protocol}",
                                ) != 0:
                                    related_status = "failed"
                                    related_reason = "candidate contract tests failed"
                                related_results.append(
                                    {
                                        "protocol": related_protocol,
                                        "idea_id": related_idea_id,
                                        "status": related_status,
                                        "speedup_over_baseline": related_speedup,
                                        "baseline_result": related_baseline_result,
                                        "candidate_result": related_candidate_result,
                                        "artifacts": related_artifacts,
                                    }
                                )
                                related_result_appended = True
                                if related_status in {"won", "neutral"}:
                                    comparison_paths.append(related_comparison_csv)
                            else:
                                related_reason = "missing successful related baseline or candidate result"

                        if related_reason and not related_result_appended:
                            related_results.append(
                                {
                                    "protocol": related_protocol,
                                    "idea_id": related_idea_id,
                                    "status": related_status,
                                    "reason": related_reason,
                                    "artifacts": related_artifacts,
                                }
                            )

                        record_state(
                            decision=decision,
                            label=f"record_result_related_{related_protocol}",
                            idea_id=related_idea_id,
                            protocol=related_protocol,
                            knobs_json=args.knobs_json,
                            status=related_status,
                            artifacts=related_artifacts,
                            reason=related_reason,
                        )

                    decision["related_results"] = related_results
                    all_comparison_csv = run_dir / "all_comparisons.csv"
                    combine_comparisons(comparison_paths, all_comparison_csv)
                    slurm_comparison_csv = all_comparison_csv if all_comparison_csv.exists() else comparison_csv
                    decision["all_comparisons_csv"] = str(slurm_comparison_csv)
                    if decision.get("status") == "won" and not args.no_slurm:
                        run_cmd(
                            [
                                sys.executable,
                                str(SLURM_SCRIPT),
                                "--comparison-csv",
                                str(slurm_comparison_csv),
                                "--measured-iters",
                                "20",
                            ],
                            decision,
                            "make_slurm",
                        )
                else:
                    decision.update({"status": "neutral", "reason": "candidate did not beat baseline"})
            else:
                decision.update({"status": "failed", "reason": "missing successful baseline or candidate result"})

        artifacts = {
            "run_dir": str(run_dir),
            "decision_json": str(decision_path),
            "baseline_dir": str(run_dir / "baseline"),
            "candidate_dir": str(run_dir / "candidate"),
            "comparison_csv": str(run_dir / "comparison.csv"),
        }
        evidence_paths = [
            artifacts["baseline_dir"],
            artifacts["candidate_dir"],
            artifacts["comparison_csv"],
        ]
        if decision.get("all_comparisons_csv"):
            evidence_paths.append(str(decision["all_comparisons_csv"]))
        decision["recommendation"] = make_recommendation(
            stage="local_botero",
            status=str(decision.get("status", "failed")),
            candidate=args.candidate_name,
            reason=str(decision.get("reason", "")),
            evidence_paths=evidence_paths,
            speedup=decision.get("speedup_over_baseline"),
        )
        run_cmd(
            [
                sys.executable,
                str(IDEA_STATE_SCRIPT),
                "record-result",
                "--idea-id",
                args.idea_id,
                "--protocol",
                args.protocol,
                "--knobs-json",
                args.knobs_json,
                "--status",
                str(decision.get("status", "failed")),
                "--artifacts-json",
                json.dumps(artifacts, sort_keys=True),
                "--rejection-reason",
                str(decision.get("reason", "")),
            ],
            decision,
            "record_result",
        )
        decision["artifacts"] = artifacts
        decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0 if decision.get("status") in {"won", "neutral", "skipped"} else 1
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
