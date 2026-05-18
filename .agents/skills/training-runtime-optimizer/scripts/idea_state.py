#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = ROOT / ".agents" / "skills" / "training-runtime-optimizer"
DEFAULT_STATE = SKILL_DIR / "state" / "ideas_tested.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "ideas": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    ideas = data.setdefault("ideas", [])
    if not isinstance(ideas, list):
        raise SystemExit(f"{path} field 'ideas' must be a list")
    data.setdefault("version", 1)
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_knobs(knobs_json: str) -> dict[str, Any]:
    if not knobs_json:
        return {}
    value = json.loads(knobs_json)
    if not isinstance(value, dict):
        raise SystemExit("--knobs-json must decode to a JSON object")
    return value


def idea_matches(entry: dict[str, Any], idea_id: str, protocol: str, knobs: dict[str, Any]) -> bool:
    return (
        entry.get("idea_id") == idea_id
        and entry.get("protocol") == protocol
        and entry.get("runtime_knobs", {}) == knobs
    )


def find_idea(state: dict[str, Any], idea_id: str, protocol: str, knobs: dict[str, Any]) -> dict[str, Any] | None:
    for entry in state["ideas"]:
        if isinstance(entry, dict) and idea_matches(entry, idea_id, protocol, knobs):
            return entry
    return None


def parse_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-file", default=str(DEFAULT_STATE))


def add_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--knobs-json", default="{}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Track tested runtime ideas for the training-runtime-optimizer skill")
    parse_common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Print the full idea state")
    parse_common(list_parser)

    has_parser = subparsers.add_parser("has-tested", help="Exit 0 if this idea/protocol/knob set is already recorded")
    parse_common(has_parser)
    add_identity_args(has_parser)

    started_parser = subparsers.add_parser("record-started", help="Record that an idea run started")
    parse_common(started_parser)
    add_identity_args(started_parser)
    started_parser.add_argument("--source", default="repo")
    started_parser.add_argument("--rationale", default="")
    started_parser.add_argument("--touched-behavior", default="")
    started_parser.add_argument("--source-notes", default="")

    result_parser = subparsers.add_parser("record-result", help="Record or update the final result for an idea")
    parse_common(result_parser)
    add_identity_args(result_parser)
    result_parser.add_argument("--status", required=True, choices=["started", "won", "neutral", "failed", "skipped"])
    result_parser.add_argument("--baseline-json", default="")
    result_parser.add_argument("--candidate-json", default="")
    result_parser.add_argument("--comparison-json", default="")
    result_parser.add_argument("--artifacts-json", default="{}")
    result_parser.add_argument("--rejection-reason", default="")

    skipped_parser = subparsers.add_parser("mark-skipped", help="Record a skipped idea or cycle")
    parse_common(skipped_parser)
    add_identity_args(skipped_parser)
    skipped_parser.add_argument("--reason", required=True)

    args = parser.parse_args()
    state_path = Path(args.state_file)
    state = load_state(state_path)

    if args.command == "list":
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    knobs = normalize_knobs(args.knobs_json)
    entry = find_idea(state, args.idea_id, args.protocol, knobs)

    if args.command == "has-tested":
        return 0 if entry is not None else 1

    if args.command == "record-started":
        if entry is None:
            entry = {
                "idea_id": args.idea_id,
                "protocol": args.protocol,
                "runtime_knobs": knobs,
                "created_at": now_iso(),
            }
            state["ideas"].append(entry)
        entry.update(
            {
                "updated_at": now_iso(),
                "status": "started",
                "source": args.source,
                "rationale": args.rationale,
                "touched_behavior": args.touched_behavior,
                "source_notes": args.source_notes,
            }
        )
        save_state(state_path, state)
        return 0

    if args.command == "record-result":
        if entry is None:
            entry = {
                "idea_id": args.idea_id,
                "protocol": args.protocol,
                "runtime_knobs": knobs,
                "created_at": now_iso(),
            }
            state["ideas"].append(entry)
        update: dict[str, Any] = {
            "updated_at": now_iso(),
            "status": args.status,
            "rejection_reason": args.rejection_reason,
        }
        for field in ("baseline_json", "candidate_json", "comparison_json"):
            value = getattr(args, field)
            if value:
                update[field] = value
        if args.artifacts_json:
            artifacts = json.loads(args.artifacts_json)
            if not isinstance(artifacts, dict):
                raise SystemExit("--artifacts-json must decode to a JSON object")
            update["artifacts"] = artifacts
        entry.update(update)
        save_state(state_path, state)
        return 0

    if args.command == "mark-skipped":
        if entry is None:
            entry = {
                "idea_id": args.idea_id,
                "protocol": args.protocol,
                "runtime_knobs": knobs,
                "created_at": now_iso(),
            }
            state["ideas"].append(entry)
        entry.update({"updated_at": now_iso(), "status": "skipped", "rejection_reason": args.reason})
        save_state(state_path, state)
        return 0

    print(f"Unhandled command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
