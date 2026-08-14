"""Twice-daily notifier for suspected catastrophic overfitting.

The notifier is intentionally read-only with respect to both training clusters.
For each cluster it asks the operational DB for models that are still ``running``
plus finished models whose recorded best score is below 1, then reads those
models' per-epoch ``summary.csv`` files over SSH.  Those files contain the same
clean/adversarial values printed by the end-of-epoch ``Test`` log line, but
unlike AIRCC's array stdout they are not interleaved between two processes.

The only local mutation is a small deduplication state file, written after a
successful alert.  The script never changes a job, checkpoint, or remote file.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = REPO_ROOT / "data_analysis" / "logs" / "catastrophic_notifier_state.json"
REMOTE_RESULT_MARKER = "ARES_CATASTROPHIC_JSON="

DEFAULT_CLOSE_GAP = 2.0
DEFAULT_PRIOR_COMMON_GAP = 5.0
DEFAULT_PRIOR_PEAK_GAP = 8.0
DEFAULT_LOOKBACK = 5
DEFAULT_CONFIRMATIONS = 2
RECOVERY_GAP = 5.0
FINISHED_SCORE_THRESHOLD = 1.0


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    ssh_host: str
    remote_repo: str
    db_relative_path: str


@dataclass(frozen=True)
class EpochEval:
    epoch: int
    clean_acc: float
    adv_acc: float
    adv_loss: float
    adv_top5: float

    @property
    def gap(self) -> float:
        return abs(self.clean_acc - self.adv_acc)

    @property
    def proves_adversarial_eval(self) -> bool:
        return any(abs(value) > 0 for value in (self.adv_acc, self.adv_loss, self.adv_top5))


@dataclass(frozen=True)
class CollapseEvent:
    cluster: str
    model_name: str
    possible_epoch: int
    confirmation_epoch: int
    clean_acc: float
    adv_acc: float
    gap: float
    prior_median_gap: float
    prior_peak_gap: float
    latest_epoch: int
    latest_clean_acc: float
    latest_adv_acc: float
    latest_gap: float
    recovered_epoch: Optional[int] = None
    job_status: str = "running"
    db_best_score: Optional[float] = None

    @property
    def event_id(self) -> str:
        return f"{self.cluster}:{self.model_name}:{self.possible_epoch}"


@dataclass(frozen=True)
class ModelSnapshot:
    summary: str
    job_status: str = "running"
    db_best_score: Optional[float] = None


@dataclass(frozen=True)
class ClusterSnapshot:
    models: Mapping[str, ModelSnapshot]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectorConfig:
    close_gap: float = DEFAULT_CLOSE_GAP
    prior_common_gap: float = DEFAULT_PRIOR_COMMON_GAP
    prior_peak_gap: float = DEFAULT_PRIOR_PEAK_GAP
    lookback: int = DEFAULT_LOOKBACK
    confirmations: int = DEFAULT_CONFIRMATIONS


REMOTE_COLLECTOR = r'''import json
import pathlib
import sqlite3
import sys

repo = pathlib.Path(sys.argv[1])
db_path = repo / sys.argv[2]
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)

def eligible_models():
    rows = conn.execute(
        "SELECT model_name, status, best_score FROM jobs "
        "WHERE status='running' OR "
        "(status='finished' AND best_score IS NOT NULL AND best_score < ?)",
        (1.0,),
    )
    return {
        row[0]: {"status": row[1], "best_score": row[2]}
        for row in rows
    }

initial = eligible_models()
summaries = {}
errors = []
for name in sorted(initial):
    path = repo / "results" / "models" / name / "summary.csv"
    try:
        summaries[name] = path.read_text(errors="replace")
    except FileNotFoundError:
        pass  # A newly claimed model may not have completed its first eval yet.
    except OSError as exc:
        errors.append(f"{name}: cannot read {path}: {exc}")

# A lifecycle can finish while files are being read. Recheck eligibility so a
# normally finished model disappears, while one finishing below score 1 stays.
final = eligible_models()
still_eligible = set(initial) & set(final)
payload = {
    "models": {
        name: {
            "summary": summaries[name],
            "status": final[name]["status"],
            "best_score": final[name]["best_score"],
        }
        for name in sorted(still_eligible)
        if name in summaries
    },
    "errors": errors,
}
print("ARES_CATASTROPHIC_JSON=" + json.dumps(payload, separators=(",", ":")))
'''


def default_clusters() -> dict[str, ClusterConfig]:
    return {
        "aircc": ClusterConfig(
            name="aircc",
            ssh_host=os.environ.get("CATASTROPHIC_AIRCC_HOST", "aircc"),
            remote_repo=os.environ.get(
                "CATASTROPHIC_AIRCC_REPO",
                "/shared/cycle2_bgu_golan_prj/ashtomer/ares",
            ),
            db_relative_path=os.environ.get(
                "CATASTROPHIC_AIRCC_DB",
                "aircc/aircc_job_manager/aircc_jobs.sqlite",
            ),
        ),
        "slurm": ClusterConfig(
            name="slurm",
            ssh_host=os.environ.get("CATASTROPHIC_SLURM_HOST", "slurm"),
            remote_repo=os.environ.get(
                "CATASTROPHIC_SLURM_REPO",
                "/home/ashtomer/projects/ares",
            ),
            db_relative_path=os.environ.get(
                "CATASTROPHIC_SLURM_DB",
                "slurm_job_manager/jobs.sqlite",
            ),
        ),
    }


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value}")
    return number


def parse_summary(text: str) -> list[EpochEval]:
    """Parse, sort, and de-duplicate end-of-epoch summary rows.

    Resumed jobs can append another CSV header or repeat an epoch.  The newest
    valid occurrence of an epoch wins.  A concurrently written partial final
    row is ignored.
    """

    rows_by_epoch: dict[int, EpochEval] = {}
    header: Optional[dict[str, int]] = None
    required = ("epoch", "eval_top1", "eval_advtop1")

    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        normalized = [cell.strip() for cell in row]
        if normalized[0] == "epoch":
            candidate = {name: idx for idx, name in enumerate(normalized)}
            if all(name in candidate for name in required):
                header = candidate
            continue
        if header is None:
            continue
        try:
            epoch = int(_finite_float(normalized[header["epoch"]]))
            clean = _finite_float(normalized[header["eval_top1"]])
            adv = _finite_float(normalized[header["eval_advtop1"]])
            adv_loss = _finite_float(normalized[header["eval_advloss"]]) if "eval_advloss" in header else 0.0
            adv_top5 = _finite_float(normalized[header["eval_advtop5"]]) if "eval_advtop5" in header else 0.0
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        rows_by_epoch[epoch] = EpochEval(epoch, clean, adv, adv_loss, adv_top5)

    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def _first_recovery(rows: Sequence[EpochEval], start: int) -> Optional[int]:
    for idx in range(start, len(rows) - 1):
        first, second = rows[idx], rows[idx + 1]
        if second.epoch != first.epoch + 1:
            continue
        if first.gap >= RECOVERY_GAP and second.gap >= RECOVERY_GAP:
            return first.epoch
    return None


def detect_collapses(
    cluster: str,
    model_name: str,
    rows: Sequence[EpochEval],
    config: DetectorConfig = DetectorConfig(),
    *,
    job_status: str = "running",
    db_best_score: Optional[float] = None,
) -> list[CollapseEvent]:
    """Return distinct sustained close-gap events in one model's history."""

    if not rows or not any(row.proves_adversarial_eval for row in rows):
        return []
    if config.lookback < 1 or config.confirmations < 1:
        raise ValueError("lookback and confirmations must be positive")

    events: list[CollapseEvent] = []
    idx = config.lookback
    last_start = len(rows) - config.confirmations
    while idx <= last_start:
        confirmations = rows[idx : idx + config.confirmations]
        consecutive = all(
            confirmations[offset].epoch == confirmations[0].epoch + offset
            for offset in range(len(confirmations))
        )
        close = all(row.gap <= config.close_gap for row in confirmations)
        prior = rows[idx - config.lookback : idx]
        prior_gaps = [row.gap for row in prior]
        drastic = (
            sum(gap >= config.prior_common_gap for gap in prior_gaps) >= 2
            and max(prior_gaps) >= config.prior_peak_gap
        )
        if not (consecutive and close and drastic):
            idx += 1
            continue

        possible = confirmations[0]
        confirmed = confirmations[-1]
        latest = rows[-1]
        events.append(
            CollapseEvent(
                cluster=cluster,
                model_name=model_name,
                possible_epoch=possible.epoch,
                confirmation_epoch=confirmed.epoch,
                clean_acc=possible.clean_acc,
                adv_acc=possible.adv_acc,
                gap=possible.gap,
                prior_median_gap=statistics.median(prior_gaps),
                prior_peak_gap=max(prior_gaps),
                latest_epoch=latest.epoch,
                latest_clean_acc=latest.clean_acc,
                latest_adv_acc=latest.adv_acc,
                latest_gap=latest.gap,
                recovered_epoch=_first_recovery(rows, idx + config.confirmations),
                job_status=job_status,
                db_best_score=db_best_score,
            )
        )

        # Treat the whole contiguous close-gap stretch as this event.  A later
        # recurrence is eligible only after the series has moved away again.
        idx += config.confirmations
        while (
            idx < len(rows)
            and rows[idx].epoch == rows[idx - 1].epoch + 1
            and rows[idx].gap <= config.close_gap
        ):
            idx += 1

    return events


def collect_cluster(config: ClusterConfig, timeout: int = 180) -> ClusterSnapshot:
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        config.ssh_host,
        "python3",
        "-",
        config.remote_repo,
        config.db_relative_path,
    ]
    try:
        proc = subprocess.run(
            command,
            input=REMOTE_COLLECTOR,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{config.name}: SSH collector failed: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-2000:]
        raise RuntimeError(
            f"{config.name}: SSH collector exited {proc.returncode}: {detail or '(no output)'}"
        )

    marker_lines = [line for line in proc.stdout.splitlines() if line.startswith(REMOTE_RESULT_MARKER)]
    if not marker_lines:
        raise RuntimeError(f"{config.name}: collector returned no JSON marker")
    try:
        payload = json.loads(marker_lines[-1][len(REMOTE_RESULT_MARKER) :])
        raw_models = payload["models"]
        errors = tuple(str(item) for item in payload.get("errors", []))
        if not isinstance(raw_models, dict):
            raise TypeError("models is not a mapping")
        models: dict[str, ModelSnapshot] = {}
        for name, raw in raw_models.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise TypeError("invalid model entry")
            summary = raw.get("summary")
            status = raw.get("status")
            best_score = raw.get("best_score")
            if not isinstance(summary, str) or status not in ("running", "finished"):
                raise TypeError(f"invalid model metadata for {name}")
            if best_score is not None:
                best_score = float(best_score)
            if status == "finished" and (best_score is None or best_score >= FINISHED_SCORE_THRESHOLD):
                raise TypeError(f"ineligible finished model returned for {name}")
            models[name] = ModelSnapshot(summary, status, best_score)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{config.name}: invalid collector JSON: {exc}") from exc
    return ClusterSnapshot(models=models, errors=errors)


def scan_clusters(
    clusters: Iterable[ClusterConfig],
    detector: DetectorConfig,
    collector: Callable[[ClusterConfig], ClusterSnapshot] = collect_cluster,
) -> tuple[list[CollapseEvent], list[str], dict[str, dict[str, int]]]:
    events: list[CollapseEvent] = []
    errors: list[str] = []
    selection_counts: dict[str, dict[str, int]] = {}
    for cluster in clusters:
        try:
            snapshot = collector(cluster)
        except Exception as exc:
            errors.append(str(exc))
            continue
        selection_counts[cluster.name] = {
            "running": sum(model.job_status == "running" for model in snapshot.models.values()),
            "finished_score_under_1": sum(
                model.job_status == "finished" for model in snapshot.models.values()
            ),
        }
        errors.extend(f"{cluster.name}: {error}" for error in snapshot.errors)
        for model_name, model in snapshot.models.items():
            rows = parse_summary(model.summary)
            events.extend(
                detect_collapses(
                    cluster.name,
                    model_name,
                    rows,
                    detector,
                    job_status=model.job_status,
                    db_best_score=model.db_best_score,
                )
            )
    events.sort(key=lambda event: (event.cluster, event.model_name, event.possible_epoch))
    return events, errors, selection_counts


def _format_selection_counts(selection_counts: Mapping[str, Mapping[str, int]]) -> str:
    parts = []
    for cluster, counts in sorted(selection_counts.items()):
        parts.append(
            f"{cluster}: running={counts.get('running', 0)}, "
            f"finished_score_under_1={counts.get('finished_score_under_1', 0)}"
        )
    return "; ".join(parts) or "none"


def format_event_report(
    events: Sequence[CollapseEvent],
    selection_counts: Mapping[str, Mapping[str, int]],
) -> str:
    unique_models = {(event.cluster, event.model_name) for event in events}
    lines = [
        f"Suspected catastrophic overfitting: {len(events)} event(s) across "
        f"{len(unique_models)} eligible model(s).",
        "Detection uses sorted end-of-epoch summary metrics from the cluster logs.",
        "",
    ]
    for event in events:
        selection = event.job_status
        if event.job_status == "finished":
            selection += f", DB best_score={event.db_best_score:.6g}"
        recovery = (
            f"recovered by epoch {event.recovered_epoch}"
            if event.recovered_epoch is not None
            else "not yet recovered in the available evaluations"
        )
        lines.extend(
            [
                f"[{event.cluster}] {event.model_name} ({selection})",
                f"  possible collapse epoch: {event.possible_epoch} "
                f"(confirmed at {event.confirmation_epoch})",
                f"  at detection: clean={event.clean_acc:.3f}, adv={event.adv_acc:.3f}, "
                f"gap={event.gap:.3f} points",
                f"  preceding window: median gap={event.prior_median_gap:.3f}, "
                f"peak gap={event.prior_peak_gap:.3f} points",
                f"  latest epoch {event.latest_epoch}: clean={event.latest_clean_acc:.3f}, "
                f"adv={event.latest_adv_acc:.3f}, gap={event.latest_gap:.3f}; {recovery}",
                "",
            ]
        )
    lines.append(
        "Eligible models with readable epoch summaries: "
        + _format_selection_counts(selection_counts)
    )
    return "\n".join(lines).rstrip() + "\n"


def format_scan_errors(
    errors: Sequence[str],
    selection_counts: Mapping[str, Mapping[str, int]],
) -> str:
    lines = ["The catastrophic-overfitting scan was incomplete.", ""]
    lines.extend(f"- {error}" for error in errors)
    lines.extend(
        [
            "",
            "Successfully collected readable summaries: "
            + _format_selection_counts(selection_counts),
        ]
    )
    return "\n".join(lines) + "\n"


def load_reported_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text())
        reported = payload.get("reported", [])
        if not isinstance(reported, list) or not all(isinstance(item, str) for item in reported):
            raise ValueError("reported must be a list of strings")
        return set(reported)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read notifier state {path}: {exc}") from exc


def save_reported_ids(path: Path, reported: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reported": sorted(reported),
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def notifier_lock(state_path: Path) -> Iterator[bool]:
    lock_path = Path(f"{state_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _make_emailer():
    from aircc.aircc_job_manager.notify import make_emailer

    return make_emailer()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster",
        action="append",
        choices=("aircc", "slurm"),
        help="Cluster to inspect; repeat for both. Defaults to both.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print findings; do not email or write state.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--close-gap", type=float, default=DEFAULT_CLOSE_GAP)
    parser.add_argument("--prior-common-gap", type=float, default=DEFAULT_PRIOR_COMMON_GAP)
    parser.add_argument("--prior-peak-gap", type=float, default=DEFAULT_PRIOR_PEAK_GAP)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--confirmations", type=int, default=DEFAULT_CONFIRMATIONS)
    return parser


def run(
    args: argparse.Namespace,
    *,
    collector: Callable[[ClusterConfig], ClusterSnapshot] = collect_cluster,
    emailer_factory: Callable[[], Optional[Callable[[str, str], None]]] = _make_emailer,
) -> int:
    configs = default_clusters()
    selected = args.cluster or ["aircc", "slurm"]
    clusters = [configs[name] for name in selected]
    detector = DetectorConfig(
        close_gap=args.close_gap,
        prior_common_gap=args.prior_common_gap,
        prior_peak_gap=args.prior_peak_gap,
        lookback=args.lookback,
        confirmations=args.confirmations,
    )
    if detector.close_gap < 0 or detector.prior_common_gap < 0 or detector.prior_peak_gap < 0:
        raise ValueError("gap thresholds must be non-negative")

    events, errors, selection_counts = scan_clusters(clusters, detector, collector)
    if args.dry_run:
        print(format_event_report(events, selection_counts) if events else "No collapse events detected.")
        if errors:
            print(format_scan_errors(errors, selection_counts), file=sys.stderr)
        return 1 if errors else 0

    try:
        reported = load_reported_ids(args.state_file)
    except RuntimeError as exc:
        errors.append(str(exc))
        reported = set()
        state_usable = False
    else:
        state_usable = True

    new_events = [event for event in events if event.event_id not in reported]
    needs_email = bool(new_events or errors)
    emailer = emailer_factory() if needs_email else None
    if needs_email and emailer is None:
        if new_events:
            print(format_event_report(new_events, selection_counts), file=sys.stderr)
        if errors:
            print(format_scan_errors(errors, selection_counts), file=sys.stderr)
        print("No email transport is configured; alert was not recorded and will be retried.", file=sys.stderr)
        return 1

    if new_events:
        assert emailer is not None
        unique_models = {(event.cluster, event.model_name) for event in new_events}
        try:
            emailer(
                f"[ares] catastrophic overfitting suspected: {len(new_events)} event(s), "
                f"{len(unique_models)} model(s)",
                format_event_report(new_events, selection_counts),
            )
        except Exception as exc:
            print(f"Catastrophic-overfitting email failed: {exc}", file=sys.stderr)
            return 1
        if state_usable:
            reported.update(event.event_id for event in new_events)
            try:
                save_reported_ids(args.state_file, reported)
            except OSError as exc:
                print(f"Alert sent but notifier state could not be saved: {exc}", file=sys.stderr)
                return 1

    if errors:
        assert emailer is not None
        try:
            emailer(
                "[ares] catastrophic notifier scan incomplete",
                format_scan_errors(errors, selection_counts),
            )
        except Exception as exc:
            print(f"Scan-incomplete email failed: {exc}", file=sys.stderr)
        return 1

    if new_events:
        print(f"Sent catastrophic-overfitting alert for {len(new_events)} new event(s).")
    else:
        print("No new catastrophic-overfitting events.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return run(args)
    with notifier_lock(args.state_file) as acquired:
        if not acquired:
            print("Another catastrophic-overfitting notifier pass is still running; skipping.")
            return 0
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
