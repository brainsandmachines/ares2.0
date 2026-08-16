from __future__ import annotations

from aircc.aircc_job_manager import notify


def test_clamp_body_leaves_a_short_body_alone():
    assert notify._clamp_body("all fine\n") == "all fine\n"


def test_clamp_body_splits_carriage_return_redraws_into_lines():
    # rsync --info=progress2 redraws one "line" with \r; unsplit, the tail we keep
    # would be a single unreadable smear.
    body = "\r".join(f"{i:>12} 18%  3.50MB/s" for i in range(5))
    assert notify._clamp_body(body).splitlines() == body.split("\r")


def test_clamp_body_head_and_tails_an_oversized_body():
    body = "HEAD" + ("x" * 200_000) + "TAIL"
    out = notify._clamp_body(body, limit=1000)
    assert len(out) < 1200          # the limit plus the trim marker
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "characters trimmed" in out


def test_clamp_subject_is_one_bounded_line():
    assert notify._clamp_subject("[aircc] backup\nrsync failed") == "[aircc] backup rsync failed"
    assert len(notify._clamp_subject("x" * 500)) == notify.MAX_SUBJECT_CHARS


# --- alert spooling -------------------------------------------------------

import json
from datetime import datetime, timedelta

import pytest


def test_spool_path_is_off_unless_the_env_var_is_set(monkeypatch):
    monkeypatch.delenv(notify.SPOOL_ENV, raising=False)
    assert notify.spool_path() is None
    monkeypatch.setenv(notify.SPOOL_ENV, "0")
    assert notify.spool_path() is None


def test_spool_path_accepts_the_flag_or_an_explicit_path(monkeypatch, tmp_path):
    monkeypatch.setenv(notify.SPOOL_ENV, "1")
    assert notify.spool_path() == notify.DEFAULT_SPOOL_PATH
    monkeypatch.setenv(notify.SPOOL_ENV, str(tmp_path / "alerts.jsonl"))
    assert notify.spool_path() == tmp_path / "alerts.jsonl"


def test_append_spool_writes_one_clamped_json_record_per_line(tmp_path):
    spool = tmp_path / "alerts.jsonl"
    notify.append_spool(spool, source="aircc.backup", subject="a\nb",
                        body="x" * 50_000, dedup_key="k1")
    notify.append_spool(spool, source="sjm.status", subject="second", body="short")

    records = [json.loads(line) for line in spool.read_text().splitlines()]
    assert [r["subject"] for r in records] == ["a b", "second"]
    assert records[0]["dedup_key"] == "k1"
    assert "dedup_key" not in records[1]
    assert len(records[0]["body"]) < notify.MAX_BODY_CHARS + 200
    assert "characters trimmed" in records[0]["body"]


def _fake_transport(sent):
    def _send(subject, body):
        sent.append((subject, body))
    return _send


def test_spoolable_spools_normal_alerts_and_mails_urgent_ones(monkeypatch, tmp_path):
    spool = tmp_path / "alerts.jsonl"
    monkeypatch.setenv(notify.SPOOL_ENV, str(spool))
    sent = []
    emit = notify._spoolable(_fake_transport(sent), "aircc.daily_monitor")

    emit("normal one", "body", dedup_key="k")
    assert sent == []
    emit("urgent one", "body", urgent=True)
    assert [s for s, _ in sent] == ["urgent one"]

    records = [json.loads(line) for line in spool.read_text().splitlines()]
    assert [r["subject"] for r in records] == ["normal one"]
    assert records[0]["source"] == "aircc.daily_monitor"


def test_spoolable_mails_everything_when_no_spool_is_configured(monkeypatch):
    monkeypatch.delenv(notify.SPOOL_ENV, raising=False)
    sent = []
    emit = notify._spoolable(_fake_transport(sent), "aircc.backup")
    emit("subject", "body", dedup_key="k")
    assert [s for s, _ in sent] == ["subject"]


def _write_spool(spool, ts):
    spool.write_text(json.dumps({"ts": ts, "source": "s", "subject": "x", "body": ""}) + "\n")


def test_spool_is_stale_only_once_the_oldest_entry_ages_out(tmp_path):
    spool = tmp_path / "alerts.jsonl"
    now = datetime.now().astimezone()
    assert notify.spool_is_stale(spool) is False          # no file yet
    _write_spool(spool, (now - timedelta(hours=2)).isoformat())
    assert notify.spool_is_stale(spool) is False
    _write_spool(spool, (now - timedelta(hours=99)).isoformat())
    assert notify.spool_is_stale(spool) is True


def test_a_stale_spool_falls_back_to_mailing_immediately(monkeypatch, tmp_path):
    # If the digest cron dies, alerts must not pile up in a file nobody reads.
    spool = tmp_path / "alerts.jsonl"
    _write_spool(spool, (datetime.now().astimezone() - timedelta(hours=99)).isoformat())
    monkeypatch.setenv(notify.SPOOL_ENV, str(spool))
    sent = []
    notify._spoolable(_fake_transport(sent), "aircc.backup")("subject", "body")
    assert [s for s, _ in sent] == ["subject"]


def test_an_unparseable_spool_counts_as_stale(tmp_path):
    spool = tmp_path / "alerts.jsonl"
    spool.write_text("not json at all\n")
    assert notify.spool_is_stale(spool) is True
