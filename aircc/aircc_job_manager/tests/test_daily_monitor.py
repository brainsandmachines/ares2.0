from __future__ import annotations

import subprocess
from pathlib import Path

from aircc.aircc_job_manager import daily_monitor
from aircc.aircc_job_manager.db import AirccDB


GOOD_BACKUP = """[backup] 2026-07-01T03:00:02+03:00 rsync /src/ -> /dest/

Number of files: 10 (reg: 8, dir: 2)
Number of created files: 0
Number of deleted files: 0
Number of regular files transferred: 0
Total file size: 3,216,756,040 bytes
Total transferred file size: 0 bytes
Literal data: 0 bytes
Matched data: 0 bytes
File list size: 0
File list generation time: 0.142 seconds
File list transfer time: 0.000 seconds
Total bytes sent: 254
Total bytes received: 13

sent 254 bytes  received 13 bytes  534.00 bytes/sec
total size is 3,216,756,040  speedup is 12,047,775.43
[backup] 2026-07-01T03:00:02+03:00 done
"""


def _write_log(tmp_path: Path, text: str = GOOD_BACKUP) -> Path:
    p = tmp_path / "backup.log"
    p.write_text(text)
    return p


def _db(tmp_path: Path, running: int = 32, failed: bool = False) -> Path:
    path = tmp_path / "aircc_jobs.sqlite"
    db = AirccDB(str(path))
    for i in range(running):
        db.upsert_pending(f"run_{i:02d}", 10, i)
        db.claim_next(i, {})
    if failed:
        db.upsert_pending("bad_model", 10, 999)
        db.mark_failed("bad_model", "RuntimeError: test failure")
    return path


def test_check_backup_log_accepts_latest_done_block(tmp_path):
    log = _write_log(tmp_path)
    out = daily_monitor.check_backup_log(log, backup_rc=0)
    assert out.ok
    assert "Number of files" in out.block


def test_check_backup_log_rejects_nonzero_rc(tmp_path):
    log = _write_log(tmp_path)
    out = daily_monitor.check_backup_log(log, backup_rc=23)
    assert not out.ok
    assert "rc=23" in out.message


def test_check_backup_log_rejects_incomplete_latest_block(tmp_path):
    log = _write_log(
        tmp_path,
        GOOD_BACKUP + "\n[backup] 2026-07-02T03:00:02+03:00 rsync /src/ -> /dest/\n",
    )
    out = daily_monitor.check_backup_log(log, backup_rc=0)
    assert not out.ok
    assert "missing" in out.message


def test_check_db_accepts_expected_running_and_no_failures(tmp_path):
    out = daily_monitor.check_db(_db(tmp_path), expected_running=32)
    assert out.ok
    assert out.running_count == 32
    assert out.failed_jobs == []


def test_check_db_reports_running_mismatch(tmp_path):
    out = daily_monitor.check_db(_db(tmp_path, running=31), expected_running=32)
    assert not out.ok
    assert "running=31" in out.message


def test_run_once_sends_failure_report_and_health_email(tmp_path):
    emails = []

    def emailer(subject, body):
        emails.append((subject, body))

    def llm(prompt):
        assert "bad_model" in prompt
        return "# Suggested fix\nchange the config"

    rec = tmp_path / "recommendations"
    rc = daily_monitor.run_once(
        db_path=_db(tmp_path, failed=True),
        backup_log=_write_log(tmp_path),
        expected_running=32,
        recommendations_dir=rec,
        mount_repo=tmp_path / "mount",
        emailer=emailer,
        llm_client=llm,
    )

    assert rc == 1
    reports = list(rec.glob("bad_model.*.md"))
    assert len(reports) == 1
    assert "Suggested fix" in reports[0].read_text()
    assert any("DB health problem" in subject for subject, _ in emails)
    failure_email = [body for subject, body in emails if "failure on bad_model" in subject][0]
    assert "there is a failure on bad_model" in failure_email
    assert str(reports[0]) in failure_email


def test_run_once_writes_fallback_report_when_llm_missing(tmp_path):
    emails = []

    rec = tmp_path / "recommendations"
    rc = daily_monitor.run_once(
        db_path=_db(tmp_path, failed=True),
        backup_log=_write_log(tmp_path),
        expected_running=32,
        recommendations_dir=rec,
        mount_repo=tmp_path / "mount",
        emailer=lambda subject, body: emails.append((subject, body)),
        llm_client=None,
    )

    assert rc == 1
    reports = list(rec.glob("bad_model.*.md"))
    assert len(reports) == 1
    assert "AIRCC failure report" in reports[0].read_text()
    assert emails


def test_repeated_failure_reuses_existing_report_without_llm_call(tmp_path):
    rec = tmp_path / "recommendations"
    db_path = _db(tmp_path, failed=True)
    log_path = _write_log(tmp_path)
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return "# fix\nfirst report"

    kwargs = dict(
        db_path=db_path,
        backup_log=log_path,
        expected_running=32,
        recommendations_dir=rec,
        mount_repo=tmp_path / "mount",
        emailer=lambda _subject, _body: None,
    )
    assert daily_monitor.run_once(**kwargs, llm_client=llm) == 1
    assert daily_monitor.run_once(**kwargs, llm_client=llm) == 1

    assert len(calls) == 1
    reports = list(rec.glob("bad_model.*.md"))
    assert len(reports) == 1


def _db_with_missing_best(tmp_path: Path) -> Path:
    """A finished row whose in-training best-checkpoint write was lost."""
    path = tmp_path / "aircc_jobs.sqlite"
    db = AirccDB(str(path))
    db.upsert_pending("dep_model", 200, 0)
    db.claim_next(1, {})
    db.mark_finished("dep_model")          # hook's set_best_checkpoint never landed
    db.upsert_pending("ok_model", 200, 1)
    db.claim_next(2, {})
    db.mark_finished("ok_model")
    db.set_best_checkpoint("ok_model", "/m/ok_model/last.pth.tar", 55.0)
    return path


def test_find_missing_best_checkpoints_flags_only_the_broken_row(tmp_path):
    found = daily_monitor.find_missing_best_checkpoints(_db_with_missing_best(tmp_path))
    assert found == ["dep_model"]


def test_find_missing_best_checkpoints_is_quiet_when_all_rows_are_healthy(tmp_path):
    assert daily_monitor.find_missing_best_checkpoints(_db(tmp_path)) == []


def test_repair_runs_the_backfill_on_aircc_in_a_login_shell(tmp_path):
    seen = {}

    def runner(argv):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="wrote 1 row(s)", stderr="")

    out = daily_monitor.repair_missing_best_checkpoints(
        ["dep_model"], ssh_host="aircc", remote_repo="/shared/ares", runner=runner)

    assert out.ok and "1 flagged row(s): wrote 1 row(s)" in out.message
    argv = seen["argv"]
    assert argv[0] == "ssh" and argv[3] == "aircc"
    # one argv item, login shell, --apply -- the mount is read-only, so the write
    # has to happen remotely
    assert len(argv) == 5
    assert argv[4].startswith("bash -lc ")
    assert "cd /shared/ares" in argv[4] and "--apply" in argv[4]


def test_repair_is_a_noop_without_broken_rows(tmp_path):
    def runner(argv):  # pragma: no cover - must never be reached
        raise AssertionError("should not ssh when there is nothing to repair")

    assert daily_monitor.repair_missing_best_checkpoints([], runner=runner).ok


def test_repair_failure_is_reported_not_raised(tmp_path):
    def runner(argv):
        return subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: connect failed")

    out = daily_monitor.repair_missing_best_checkpoints(["dep_model"], runner=runner)
    assert not out.ok
    assert "rc=255" in out.message and "connect failed" in out.output


def test_run_once_repairs_and_emails_the_backfill_result(tmp_path):
    emails = []
    calls = []

    def runner(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="wrote 1 row(s)", stderr="")

    rc = daily_monitor.run_once(
        db_path=_db_with_missing_best(tmp_path),
        backup_log=_write_log(tmp_path),
        expected_running=0,
        recommendations_dir=tmp_path / "rec",
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b: emails.append((s, b)),
        runner=runner,
    )

    assert rc == 0
    assert len(calls) == 1
    subject, body = [e for e in emails if "backfill" in e[0]][0]
    assert "ok" in subject
    assert "dep_model" in body and "wrote 1 row(s)" in body


def test_run_once_only_reports_when_repair_is_disabled(tmp_path):
    emails = []

    def runner(argv):  # pragma: no cover - must never be reached
        raise AssertionError("--no-repair must not ssh")

    rc = daily_monitor.run_once(
        db_path=_db_with_missing_best(tmp_path),
        backup_log=_write_log(tmp_path),
        expected_running=0,
        recommendations_dir=tmp_path / "rec",
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b: emails.append((s, b)),
        repair=False,
        runner=runner,
    )

    assert rc == 1
    subject, body = [e for e in emails if "no best checkpoint" in e[0]][0]
    assert "dep_model" in body and "--apply" in body
