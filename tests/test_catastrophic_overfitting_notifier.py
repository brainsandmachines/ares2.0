from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from data_analysis import catastrophic_overfitting_notifier as notifier


HEADER = "epoch,eval_top1,eval_advtop1,eval_advloss,eval_advtop5\n"


def _summary(rows):
    return HEADER + "".join(
        f"{epoch},{clean},{adv},{adv_loss},{adv_top5}\n"
        for epoch, clean, adv, adv_loss, adv_top5 in rows
    )


def _rows_from_gaps(start_epoch, gaps, *, clean=70.0):
    return [
        (start_epoch + offset, clean, clean - gap, 1.0, 80.0)
        for offset, gap in enumerate(gaps)
    ]


def _snapshot(model_name, summary, *, status="running", best_score=None):
    return notifier.ClusterSnapshot(
        {
            model_name: notifier.ModelSnapshot(
                summary=summary,
                job_status=status,
                db_best_score=best_score,
            )
        }
    )


@pytest.mark.parametrize(
    ("model_name", "possible_epoch", "gaps"),
    [
        (
            "convnext_base_dvd_b_l2_2_init0",
            151,
            [7.408, 6.842, 8.548, 5.718, 2.716, 1.434, 0.592],
        ),
        (
            "convnext_base_dvd_b_l2_4_init0",
            103,
            [9.896, 10.246, 9.224, 5.306, 2.458, 1.508, 0.312],
        ),
        (
            "convnext_base_gradnorm_l1_6_init0",
            105,
            [31.240, 25.318, 21.236, 24.540, 2.724, 0.252, 0.150],
        ),
        (
            "convnext_base_l2_4_init1",
            35,
            [5.712, 4.326, 16.376, 9.656, 2.842, 0.982, 1.298],
        ),
    ],
)
def test_today_aircc_reference_failures_are_detected(model_name, possible_epoch, gaps):
    rows = notifier.parse_summary(_summary(_rows_from_gaps(possible_epoch - 5, gaps)))

    events = notifier.detect_collapses("aircc", model_name, rows)

    assert [(event.model_name, event.possible_epoch, event.confirmation_epoch) for event in events] == [
        (model_name, possible_epoch, possible_epoch + 1)
    ]


def test_parser_ignores_repeated_headers_partial_rows_and_keeps_latest_duplicate_epoch():
    text = (
        HEADER
        + "1,60,50,2,70\n"
        + HEADER
        + "1,61,59,1,75\n"
        + "2,62,60,1,76\n"
        + "3,broken"
    )

    rows = notifier.parse_summary(text)

    assert [(row.epoch, row.clean_acc, row.adv_acc) for row in rows] == [
        (1, 61.0, 59.0),
        (2, 62.0, 60.0),
    ]


def test_single_close_epoch_does_not_alert():
    rows = notifier.parse_summary(
        _summary(_rows_from_gaps(10, [12, 11, 10, 9, 8, 1, 9]))
    )

    assert notifier.detect_collapses("aircc", "noisy", rows) == []


def test_clean_only_placeholder_columns_are_not_treated_as_adversarial_eval():
    rows = [
        (epoch, clean, 0.0, 0.0, 0.0)
        for epoch, clean in enumerate([12, 11, 10, 9, 8, 1, 0.5])
    ]

    assert notifier.detect_collapses(
        "aircc", "baseline", notifier.parse_summary(_summary(rows))
    ) == []


def test_real_adversarial_run_can_collapse_to_zero_accuracy():
    rows = _rows_from_gaps(20, [12, 11, 10, 9, 8])
    rows += [(25, 1.0, 0.0, 5.0, 0.0), (26, 0.5, 0.0, 5.0, 0.0)]

    events = notifier.detect_collapses(
        "aircc", "zero_adv", notifier.parse_summary(_summary(rows))
    )

    assert [event.possible_epoch for event in events] == [25]


def test_recovery_and_later_recurrence_are_distinct_events():
    gaps = [12, 11, 10, 9, 8, 1, 1, 6, 7, 9, 10, 8, 1, 1]
    rows = notifier.parse_summary(_summary(_rows_from_gaps(1, gaps)))

    events = notifier.detect_collapses("slurm", "recurrent", rows)

    assert [event.possible_epoch for event in events] == [6, 13]
    assert events[0].recovered_epoch == 8


def test_report_distinguishes_multiple_events_from_unique_models():
    gaps = [12, 11, 10, 9, 8, 1, 1, 6, 7, 9, 10, 8, 1, 1]
    rows = notifier.parse_summary(_summary(_rows_from_gaps(1, gaps)))
    events = notifier.detect_collapses("aircc", "recurrent", rows)

    report = notifier.format_event_report(
        events,
        {"aircc": {"running": 1, "finished_score_under_1": 0}},
    )

    assert "2 event(s) across 1 eligible model(s)" in report


def test_scan_only_uses_models_returned_by_running_snapshot():
    cluster = notifier.ClusterConfig("aircc", "unused", "/unused", "unused.sqlite")
    summary = _summary(_rows_from_gaps(30, [12, 11, 10, 9, 8, 1, 1]))

    events, errors, counts = notifier.scan_clusters(
        [cluster],
        notifier.DetectorConfig(),
        collector=lambda _: _snapshot("running_model", summary),
    )

    assert [event.model_name for event in events] == ["running_model"]
    assert errors == []
    assert counts == {
        "aircc": {"running": 1, "finished_score_under_1": 0}
    }


def test_finished_model_with_db_score_under_one_is_scanned_and_labeled():
    cluster = notifier.ClusterConfig("aircc", "unused", "/unused", "unused.sqlite")
    summary = _summary(_rows_from_gaps(30, [12, 11, 10, 9, 8, 1, 1]))

    events, errors, counts = notifier.scan_clusters(
        [cluster],
        notifier.DetectorConfig(),
        collector=lambda _: _snapshot(
            "finished_low_score",
            summary,
            status="finished",
            best_score=0.75,
        ),
    )
    report = notifier.format_event_report(events, counts)

    assert errors == []
    assert counts == {
        "aircc": {"running": 0, "finished_score_under_1": 1}
    }
    assert "finished_low_score (finished, DB best_score=0.75)" in report


def test_remote_collector_selects_running_and_finished_scores_strictly_under_one(tmp_path):
    repo = tmp_path / "remote_repo"
    repo.mkdir()
    db_path = repo / "jobs.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE jobs (model_name TEXT PRIMARY KEY, status TEXT, best_score REAL)"
        )
        conn.executemany(
            "INSERT INTO jobs VALUES (?, ?, ?)",
            [
                ("running", "running", None),
                ("finished_zero", "finished", 0.0),
                ("finished_near_zero", "finished", 0.999),
                ("finished_one", "finished", 1.0),
                ("finished_null", "finished", None),
                ("pending_zero", "pending", 0.0),
            ],
        )
    for model_name in ("running", "finished_zero", "finished_near_zero"):
        model_dir = repo / "results" / "models" / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "summary.csv").write_text(
            _summary(_rows_from_gaps(30, [12, 11, 10, 9, 8, 1, 1]))
        )

    proc = subprocess.run(
        [sys.executable, "-", str(repo), "jobs.sqlite"],
        input=notifier.REMOTE_COLLECTOR,
        text=True,
        capture_output=True,
        check=True,
    )
    marker = next(
        line for line in proc.stdout.splitlines()
        if line.startswith(notifier.REMOTE_RESULT_MARKER)
    )
    payload = json.loads(marker[len(notifier.REMOTE_RESULT_MARKER) :])

    assert set(payload["models"]) == {
        "running",
        "finished_zero",
        "finished_near_zero",
    }
    assert payload["models"]["finished_zero"]["best_score"] == 0.0
    assert payload["models"]["finished_near_zero"]["best_score"] == 0.999


def test_failed_email_does_not_record_event(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    summary = _summary(_rows_from_gaps(30, [12, 11, 10, 9, 8, 1, 1]))
    args = notifier.build_parser().parse_args(["--cluster", "aircc", "--state-file", str(state)])

    def collector(_):
        return _snapshot("bad_model", summary)

    def emailer(_subject, _body):
        raise RuntimeError("smtp down")

    rc = notifier.run(args, collector=collector, emailer_factory=lambda: emailer)

    assert rc == 1
    assert not state.exists()


def test_successful_email_records_event_and_deduplicates_next_run(tmp_path):
    state = tmp_path / "state.json"
    summary = _summary(_rows_from_gaps(30, [12, 11, 10, 9, 8, 1, 1]))
    args = notifier.build_parser().parse_args(["--cluster", "aircc", "--state-file", str(state)])
    emails = []

    def collector(_):
        return _snapshot("bad_model", summary)

    def factory():
        return lambda subject, body: emails.append((subject, body))

    assert notifier.run(args, collector=collector, emailer_factory=factory) == 0
    assert notifier.run(args, collector=collector, emailer_factory=factory) == 0

    assert len(emails) == 1
    payload = json.loads(state.read_text())
    assert payload["reported"] == ["aircc:bad_model:35"]


def test_dry_run_sends_nothing_and_writes_no_state(tmp_path):
    state = tmp_path / "state.json"
    summary = _summary(_rows_from_gaps(30, [12, 11, 10, 9, 8, 1, 1]))
    args = notifier.build_parser().parse_args(
        ["--cluster", "aircc", "--state-file", str(state), "--dry-run"]
    )

    rc = notifier.run(
        args,
        collector=lambda _: _snapshot("bad_model", summary),
        emailer_factory=lambda: pytest.fail("dry-run must not create an emailer"),
    )

    assert rc == 0
    assert not state.exists()
