from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from aircc.aircc_job_manager import mail_log


def _write(directory, records, name="2026-08.jsonl"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return directory


def _record(**over):
    base = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "msg_id": "abc123def456",
        "source": "aircc.backup",
        "subject": "[aircc] backup rsync failed rc=23",
        "body": "rsync said no",
        "urgent": False,
        "routed": "spooled",
    }
    base.update(over)
    return base


def test_parse_since_accepts_durations_and_iso_dates():
    now = datetime.now().astimezone()
    assert (now - mail_log.parse_since("7d")).total_seconds() == pytest.approx(7 * 86400, abs=5)
    assert (now - mail_log.parse_since("36h")).total_seconds() == pytest.approx(36 * 3600, abs=5)
    assert mail_log.parse_since("2026-08-01").year == 2026
    with pytest.raises(Exception):
        mail_log.parse_since("last tuesday")


def test_read_records_spans_files_in_order_and_skips_junk(tmp_path, capsys):
    _write(tmp_path, [_record(subject="july")], name="2026-07.jsonl")
    (tmp_path / "2026-08.jsonl").write_text(
        json.dumps(_record(subject="august")) + "\nnot json\n\n", encoding="utf-8")

    assert [r["subject"] for r in mail_log.read_records(tmp_path)] == ["july", "august"]
    assert "unparseable" in capsys.readouterr().err


def _filter(records, **kw):
    options = {"since": None, "source": None, "pattern": None,
               "urgent_only": False, "msg_id": None}
    options.update(kw)
    return [r for r in records if mail_log.matches(r, **options)]


def test_matches_filters_by_window_source_and_urgency():
    now = datetime.now().astimezone()
    old = _record(subject="old", ts=(now - timedelta(days=30)).isoformat())
    recent = _record(subject="recent")
    other = _record(subject="other", source="sjm.status", urgent=True)
    records = [old, recent, other]

    assert [r["subject"] for r in _filter(records, since=now - timedelta(days=7))] \
        == ["recent", "other"]
    assert [r["subject"] for r in _filter(records, source="sjm.")] == ["other"]
    assert [r["subject"] for r in _filter(records, urgent_only=True)] == ["other"]


def test_msg_id_lookup_ignores_the_other_filters():
    wanted = _record(subject="wanted", msg_id="ff00aa112233",
                     ts=(datetime.now().astimezone() - timedelta(days=400)).isoformat())
    records = [_record(subject="no"), wanted]
    hits = _filter(records, msg_id="ff00aa", since=datetime.now().astimezone(),
                   urgent_only=True)
    assert [r["subject"] for r in hits] == ["wanted"]


def test_grep_searches_subject_and_body():
    records = [_record(subject="quiet", body="nothing here"),
               _record(subject="loud", body="disk is full")]
    import re
    assert [r["subject"] for r in _filter(records, pattern=re.compile("FULL", re.I))] == ["loud"]


def test_main_prints_one_line_per_alert_and_a_flag_legend(tmp_path, capsys):
    _write(tmp_path, [_record(), _record(subject="urgent one", urgent=True, routed="mailed")])

    assert mail_log.main(["--dir", str(tmp_path)]) == 0
    out = capsys.readouterr()
    assert "backup rsync failed" in out.out
    assert "aircc.backup" in out.out
    assert "2 alert(s)" in out.err


def test_main_full_prints_bodies_and_flags_a_failed_send(tmp_path, capsys):
    _write(tmp_path, [_record(body="rsync exited 23", send_error="SMTPException: nope")])

    assert mail_log.main(["--dir", str(tmp_path), "--full"]) == 0
    out = capsys.readouterr().out
    assert "rsync exited 23" in out
    assert "SEND FAILED: SMTPException: nope" in out


def test_main_reports_no_match_with_a_nonzero_exit(tmp_path, capsys):
    _write(tmp_path, [_record(subject="only this")])
    assert mail_log.main(["--dir", str(tmp_path), "--grep", "absent"]) == 1
    assert "No archived alerts match" in capsys.readouterr().err


def test_main_without_an_archive_says_so(tmp_path, capsys):
    assert mail_log.main(["--dir", str(tmp_path / "missing")]) == 1
    assert "No mail archive" in capsys.readouterr().err
