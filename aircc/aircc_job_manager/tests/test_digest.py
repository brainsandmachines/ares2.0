from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from aircc.aircc_job_manager import digest, notify


def spool_with(tmp_path: Path, *records: dict) -> Path:
    spool = tmp_path / "alerts.jsonl"
    for record in records:
        notify.append_spool(
            spool,
            source=record.get("source", "aircc.daily_monitor"),
            subject=record["subject"],
            body=record.get("body", "body text"),
            dedup_key=record.get("dedup_key"),
        )
    return spool


def collecting_emailer(sent: list):
    def _factory(**kwargs):
        def _emit(subject, body, **options):
            sent.append({"subject": subject, "body": body, **kwargs, **options})
        return _emit
    return _factory


# --- claiming -------------------------------------------------------------

def test_claim_spool_renames_so_a_concurrent_writer_starts_a_fresh_file(tmp_path):
    spool = spool_with(tmp_path, {"subject": "[aircc] failure on m1"})
    claimed = digest.claim_spool(spool)

    assert len(claimed) == 1
    assert claimed[0].name.endswith(".processing")
    assert not spool.exists()

    # A notifier writing a moment later gets a clean spool, not a lost record.
    notify.append_spool(spool, source="sjm.status", subject="later", body="b")
    assert len(digest.read_records([spool])[0]) == 1


def test_claim_spool_picks_up_files_left_by_an_earlier_failed_run(tmp_path):
    spool = spool_with(tmp_path, {"subject": "[aircc] failure on m1"})
    first = digest.claim_spool(spool)
    spool_with(tmp_path, {"subject": "[sjm] status alert: old_hb=1 cpu_stuck=0"})

    claimed = digest.claim_spool(spool)
    assert first[0] in claimed
    assert len(claimed) == 2


def test_claim_spool_ignores_an_empty_spool(tmp_path):
    spool = tmp_path / "alerts.jsonl"
    spool.write_text("")
    assert digest.claim_spool(spool) == []


def test_peek_spool_does_not_take_the_spool(tmp_path):
    spool = spool_with(tmp_path, {"subject": "[aircc] failure on m1"})
    assert digest.peek_spool(spool) == [spool]
    assert spool.exists()


# --- parsing and ordering -------------------------------------------------

def test_read_records_reports_bad_lines_without_losing_good_ones(tmp_path):
    spool = spool_with(tmp_path, {"subject": "[aircc] failure on m1"})
    with open(spool, "a") as fh:
        fh.write("{not json\n")
        fh.write(json.dumps({"no": "subject"}) + "\n")

    records, problems = digest.read_records([spool])
    assert [r["subject"] for r in records] == ["[aircc] failure on m1"]
    assert len(problems) == 2


def test_records_are_ordered_most_serious_first(tmp_path):
    spool = spool_with(
        tmp_path,
        {"subject": "[aa_sweep] cluster probe failed"},
        {"subject": "[aircc] DB health problem"},
        {"subject": "[aircc] failure on m1"},
    )
    records, _ = digest.read_records([spool])
    assert [r["subject"] for r in records] == [
        "[aircc] DB health problem",
        "[aircc] failure on m1",
        "[aa_sweep] cluster probe failed",
    ]


def test_tier_of_prefers_the_specific_rule_over_the_default():
    assert digest.tier_of("[aircc] backup rsync failed rc=23") == 1
    assert digest.tier_of("[aircc] failure on convnext_base_linf_3") == 2
    assert digest.tier_of("[aircc] something nobody has a rule for") == digest.DEFAULT_TIER


# --- repeat suppression ---------------------------------------------------

def test_a_first_sighting_is_full_and_a_repeat_collapses_to_a_line(tmp_path):
    record = {"subject": "[aircc] status alert: failed=1", "dedup_key": "aircc-status:failed=1"}
    records, _ = digest.read_records([spool_with(tmp_path, record)])

    full, repeats, state = digest.partition(records, {})
    assert len(full) == 1 and repeats == []
    assert "aircc-status:failed=1" in state

    full, repeats, state = digest.partition(records, state)
    assert full == [] and len(repeats) == 1
    assert repeats[0]["first_seen"]


def test_a_condition_that_stops_appearing_drops_out_of_state(tmp_path):
    stale = {"aircc-status:failed=1": {"first_seen": "x", "last_seen": "x", "subject": "s"}}
    records, _ = digest.read_records([spool_with(tmp_path, {"subject": "[sjm] new failure signature ab (m1)"})])
    _, _, state = digest.partition(records, stale)
    assert "aircc-status:failed=1" not in state


def test_records_without_a_dedup_key_are_always_rendered_in_full(tmp_path):
    records, _ = digest.read_records([spool_with(tmp_path, {"subject": "[aircc] failure on m1"})])
    full, repeats, _ = digest.partition(records, {})
    assert len(full) == 1 and repeats == []
    full, repeats, _ = digest.partition(records, {})
    assert len(full) == 1 and repeats == []


def test_the_same_condition_twice_in_one_window_is_counted_once(tmp_path):
    twice = {"subject": "[aircc] status alert: failed=1", "dedup_key": "k"}
    records, _ = digest.read_records([spool_with(tmp_path, twice, twice)])
    full, repeats, state = digest.partition(records, {})
    assert len(full) == 1 and repeats == []
    assert list(state) == ["k"]


# --- rendering ------------------------------------------------------------

def test_render_lists_sources_still_open_items_and_full_sections(tmp_path):
    records, _ = digest.read_records([spool_with(
        tmp_path,
        {"subject": "[aircc] DB health problem", "body": "readonly database"},
        {"subject": "[aircc] status alert: failed=1", "dedup_key": "k"},
    )])
    state = {"k": {"first_seen": "2026-08-12T03:00:00+03:00", "last_seen": "x", "subject": "s"}}
    full, repeats, _ = digest.partition(records, state)
    subject, body = digest.render(full, repeats, [])

    assert "2 alert(s)" in subject
    assert "## [aircc] DB health problem" in body
    assert "readonly database" in body
    assert "Still open" in body
    assert "2026-08-12T03:00:00+03:00" in body
    # the collapsed one is a line, not a section
    assert "## [aircc] status alert" not in body


def test_render_splits_the_body_budget_so_late_sections_survive(tmp_path):
    many = [{"subject": f"[aircc] failure on m{i}", "body": "y" * 30_000} for i in range(8)]
    records, _ = digest.read_records([spool_with(tmp_path, *many)])
    full, repeats, _ = digest.partition(records, {})
    _, body = digest.render(full, repeats, [])

    for i in range(8):
        assert f"## [aircc] failure on m{i}" in body
    assert "characters trimmed" in body


# --- run_once -------------------------------------------------------------

def test_an_empty_spool_sends_nothing(tmp_path):
    sent = []
    rc = digest.run_once(spool=tmp_path / "alerts.jsonl", state_path=tmp_path / "state.json",
                         emailer_factory=collecting_emailer(sent))
    assert rc == 0 and sent == []


def test_a_successful_run_mails_once_clears_the_spool_and_saves_state(tmp_path):
    spool = spool_with(tmp_path, {"subject": "[aircc] status alert: failed=1", "dedup_key": "k"})
    state_path = tmp_path / "state.json"
    sent = []

    rc = digest.run_once(spool=spool, state_path=state_path,
                         emailer_factory=collecting_emailer(sent))

    assert rc == 0
    assert len(sent) == 1
    # The digest must never land back in the spool it just drained.
    assert sent[0]["urgent"] is True
    assert sent[0]["source"] == "ares.digest"
    assert list(tmp_path.glob("alerts.jsonl*")) == []
    assert "k" in json.loads(state_path.read_text())["reported"]


def test_a_failed_send_keeps_the_claimed_spool_for_the_next_run(tmp_path):
    spool = spool_with(tmp_path, {"subject": "[aircc] failure on m1"})

    def _exploding(**kwargs):
        def _emit(subject, body, **options):
            raise RuntimeError("smtp down")
        return _emit

    rc = digest.run_once(spool=spool, state_path=tmp_path / "state.json",
                         emailer_factory=_exploding)

    assert rc == 1
    claimed = list(tmp_path.glob("alerts.jsonl.*.processing"))
    assert len(claimed) == 1
    assert "[aircc] failure on m1" in claimed[0].read_text()


def test_no_transport_keeps_the_spool_too(tmp_path):
    spool = spool_with(tmp_path, {"subject": "[aircc] failure on m1"})
    rc = digest.run_once(spool=spool, state_path=tmp_path / "state.json",
                         emailer_factory=lambda **kwargs: None)
    assert rc == 1
    assert len(list(tmp_path.glob("alerts.jsonl.*.processing"))) == 1


def test_dry_run_prints_without_claiming_or_sending(tmp_path, capsys):
    spool = spool_with(tmp_path, {"subject": "[aircc] failure on m1"})
    sent = []
    rc = digest.run_once(spool=spool, state_path=tmp_path / "state.json",
                         dry_run=True, emailer_factory=collecting_emailer(sent))

    assert rc == 0 and sent == []
    assert spool.exists()
    assert "[aircc] failure on m1" in capsys.readouterr().out


def test_an_unreadable_state_file_resets_suppression_instead_of_dropping_alerts(tmp_path):
    spool = spool_with(tmp_path, {"subject": "[aircc] status alert: failed=1", "dedup_key": "k"})
    state_path = tmp_path / "state.json"
    state_path.write_text("{ broken")
    sent = []

    rc = digest.run_once(spool=spool, state_path=state_path,
                         emailer_factory=collecting_emailer(sent))

    assert rc == 0
    assert len(sent) == 1
    assert "## [aircc] status alert: failed=1" in sent[0]["body"]
    assert "repeat-suppression reset" in sent[0]["body"]
