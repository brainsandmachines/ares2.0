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


def test_check_backup_log_ignores_the_log_when_the_backup_was_skipped(tmp_path):
    # The in-flight block of someone else's still-running rsync: incomplete, and
    # not this run's business. rc=75 means no backup ran, so it must not alert.
    log = _write_log(
        tmp_path,
        GOOD_BACKUP + "\n[backup] 2026-07-02T03:00:02+03:00 rsync /src/ -> /dest/\n",
    )
    out = daily_monitor.check_backup_log(log, backup_rc=daily_monitor.BACKUP_SKIPPED_RC)
    assert out.ok
    assert "skipped" in out.message


def test_run_once_sends_no_backup_email_when_the_backup_was_skipped(tmp_path):
    emails = []
    rc = daily_monitor.run_once(
        db_path=_db(tmp_path),
        backup_log=_write_log(
            tmp_path,
            GOOD_BACKUP + "\n[backup] 2026-07-02T03:00:02+03:00 rsync /src/ -> /dest/\n",
        ),
        backup_rc=daily_monitor.BACKUP_SKIPPED_RC,
        expected=32,
        recommendations_dir=tmp_path / "recommendations",
        mount_repo=tmp_path / "mount",
        emailer=lambda subject, body, **_: emails.append((subject, body)),
    )
    assert rc == 0
    assert emails == []


def test_check_db_accepts_expected_running_and_no_failures(tmp_path):
    out = daily_monitor.check_db(_db(tmp_path), expected=32)
    assert out.ok
    assert out.running_count == 32
    assert out.failed_jobs == []


def test_check_db_reports_running_mismatch(tmp_path):
    out = daily_monitor.check_db(_db(tmp_path, running=31), expected=32)
    assert not out.ok
    assert "running=31" in out.message


def test_run_once_sends_failure_report_and_health_email(tmp_path):
    emails = []

    def emailer(subject, body, **_options):
        emails.append((subject, body))

    def llm(prompt):
        assert "bad_model" in prompt
        return "# Suggested fix\nchange the config"

    rec = tmp_path / "recommendations"
    rc = daily_monitor.run_once(
        db_path=_db(tmp_path, failed=True),
        backup_log=_write_log(tmp_path),
        expected=32,
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
        expected=32,
        recommendations_dir=rec,
        mount_repo=tmp_path / "mount",
        emailer=lambda subject, body, **_: emails.append((subject, body)),
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
        expected=32,
        recommendations_dir=rec,
        mount_repo=tmp_path / "mount",
        emailer=lambda _subject, _body, **_: None,
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


def test_run_once_repairs_the_backfill_without_mailing_about_it(tmp_path):
    """A repair that worked is log material: the monitor found the problem and
    already fixed it, so there is nothing for a reader to act on."""
    emails = []
    calls = []

    def runner(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="wrote 1 row(s)", stderr="")

    rc = daily_monitor.run_once(
        db_path=_db_with_missing_best(tmp_path),
        backup_log=_write_log(tmp_path),
        expected=0,
        recommendations_dir=tmp_path / "rec",
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b, **_: emails.append((s, b)),
        runner=runner,
    )

    assert rc == 0
    assert len(calls) == 1
    assert not [e for e in emails if "backfill" in e[0]]


def test_run_once_mails_when_the_backfill_repair_fails(tmp_path):
    emails = []

    def runner(argv):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="ssh: connect failed")

    rc = daily_monitor.run_once(
        db_path=_db_with_missing_best(tmp_path),
        backup_log=_write_log(tmp_path),
        expected=0,
        recommendations_dir=tmp_path / "rec",
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b, **_: emails.append((s, b)),
        runner=runner,
    )

    assert rc == 1
    subject, body = [e for e in emails if "backfill" in e[0]][0]
    assert "FAILED" in subject
    assert "dep_model" in body


def _db_with_falsely_failed(tmp_path: Path) -> Path:
    """One row that actually finished but hit a transient DB error on the final
    write (mark_failed instead of mark_finished), and one row that genuinely
    failed for an unrelated reason."""
    path = tmp_path / "aircc_jobs.sqlite"
    db = AirccDB(str(path))
    db.upsert_pending("flaky_model", 40, 0)
    db.claim_next(1, {})
    db.mark_failed("flaky_model",
                    "lifecycle crash: DB write failed after 8 retries: "
                    "unable to open database file")
    db.upsert_pending("real_failure", 40, 1)
    db.claim_next(2, {})
    db.mark_failed("real_failure", "RuntimeError: CUDA out of memory")
    return path


def test_find_falsely_failed_jobs_flags_only_transient_db_errors(tmp_path):
    found = daily_monitor.find_falsely_failed_jobs(_db_with_falsely_failed(tmp_path))
    assert found == ["flaky_model"]


def test_find_falsely_failed_jobs_is_quiet_when_all_rows_are_healthy(tmp_path):
    assert daily_monitor.find_falsely_failed_jobs(_db(tmp_path)) == []


def test_repair_falsely_failed_runs_the_reclaim_script_in_a_login_shell(tmp_path):
    seen = {}

    def runner(argv):
        seen["argv"] = argv
        return subprocess.CompletedProcess(
            argv, 0,
            stdout="reclaimed 1 row(s); 0 unresolved (left failed)\nRECLAIMED:flaky_model",
            stderr="")

    out = daily_monitor.repair_falsely_failed_jobs(
        ["flaky_model"], ssh_host="aircc", remote_repo="/shared/ares", runner=runner)

    assert out.ok and out.reclaimed == ["flaky_model"]
    assert "1 flagged row(s): reclaimed 1 row(s); 0 unresolved" in out.message
    argv = seen["argv"]
    assert argv[0] == "ssh" and argv[3] == "aircc"
    assert len(argv) == 5
    assert argv[4].startswith("bash -lc ")
    assert "cd /shared/ares" in argv[4] and "--apply" in argv[4]
    assert "reclaim_completed_failures" in argv[4]


def test_repair_falsely_failed_is_a_noop_without_candidates(tmp_path):
    def runner(argv):  # pragma: no cover - must never be reached
        raise AssertionError("should not ssh when there is nothing to reclaim")

    assert daily_monitor.repair_falsely_failed_jobs([], runner=runner).ok


def test_repair_falsely_failed_failure_is_reported_not_raised(tmp_path):
    def runner(argv):
        return subprocess.CompletedProcess(argv, 255, stdout="", stderr="ssh: connect failed")

    out = daily_monitor.repair_falsely_failed_jobs(["flaky_model"], runner=runner)
    assert not out.ok
    assert "rc=255" in out.message and "connect failed" in out.output
    assert out.reclaimed == []


def test_run_once_reclaims_and_skips_diagnosis_for_the_reclaimed_model(tmp_path):
    emails = []
    llm_calls = []

    def runner(argv):
        return subprocess.CompletedProcess(
            argv, 0,
            stdout="reclaimed 1 row(s); 0 unresolved (left failed)\nRECLAIMED:flaky_model",
            stderr="")

    def llm(prompt):
        llm_calls.append(prompt)
        return "# fix"

    rec = tmp_path / "rec"
    rc = daily_monitor.run_once(
        db_path=_db_with_falsely_failed(tmp_path),
        backup_log=_write_log(tmp_path),
        expected=0,
        recommendations_dir=rec,
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b, **_: emails.append((s, b)),
        llm_client=llm,
        runner=runner,
    )

    # real_failure is a genuine failure and still needs attention
    assert rc == 1
    assert len(llm_calls) == 1
    assert "real_failure" in llm_calls[0] and "flaky_model" not in llm_calls[0]
    reports = list(rec.glob("*.md"))
    assert len(reports) == 1
    assert reports[0].name.startswith("real_failure.")
    # the reclaim succeeded, so it stays out of the inbox
    assert not any("failed-row reclaim" in s for s, _ in emails)
    assert not any("failure on flaky_model" in s for s, _ in emails)
    assert any("failure on real_failure" in s for s, _ in emails)


def test_run_once_still_diagnoses_when_reclaim_cannot_resolve_it(tmp_path):
    def runner(argv):
        return subprocess.CompletedProcess(
            argv, 0,
            stdout="reclaimed 0 row(s); 1 unresolved (left failed)\nRECLAIMED:",
            stderr="")

    rec = tmp_path / "rec"
    rc = daily_monitor.run_once(
        db_path=_db_with_falsely_failed(tmp_path),
        backup_log=_write_log(tmp_path),
        expected=0,
        recommendations_dir=rec,
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b, **_: None,
        llm_client=lambda prompt: "# fix",
        runner=runner,
    )

    assert rc == 1
    # neither row was reclaimed, so both still go through the normal diagnosis path
    reports = sorted(p.name for p in rec.glob("*.md"))
    assert len(reports) == 2
    assert any(r.startswith("flaky_model.") for r in reports)
    assert any(r.startswith("real_failure.") for r in reports)


def test_run_once_only_reports_falsely_failed_when_repair_is_disabled(tmp_path):
    emails = []

    def runner(argv):  # pragma: no cover - must never be reached
        raise AssertionError("--no-repair must not ssh for the reclaim script either")

    rc = daily_monitor.run_once(
        db_path=_db_with_falsely_failed(tmp_path),
        backup_log=_write_log(tmp_path),
        expected=0,
        recommendations_dir=tmp_path / "rec",
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b, **_: emails.append((s, b)),
        llm_client=lambda prompt: "# fix",
        repair=False,
        runner=runner,
    )

    assert rc == 1
    subject, body = [e for e in emails if "may have actually completed" in e[0]][0]
    assert "flaky_model" in body and "--apply" in body


def test_run_once_only_reports_when_repair_is_disabled(tmp_path):
    emails = []

    def runner(argv):  # pragma: no cover - must never be reached
        raise AssertionError("--no-repair must not ssh")

    rc = daily_monitor.run_once(
        db_path=_db_with_missing_best(tmp_path),
        backup_log=_write_log(tmp_path),
        expected=0,
        recommendations_dir=tmp_path / "rec",
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b, **_: emails.append((s, b)),
        repair=False,
        runner=runner,
    )

    assert rc == 1
    subject, body = [e for e in emails if "no best checkpoint" in e[0]][0]
    assert "dep_model" in body and "--apply" in body


def test_expected_running_is_derived_from_squeue_not_hardcoded(tmp_path):
    seen = {}

    def runner(argv):
        seen["argv"] = argv
        # 3 RUNNING array tasks -> 3 x N_SLOTS training processes
        return subprocess.CompletedProcess(argv, 0, stdout="153201_1\n153201_2\n153201_3\n", stderr="")

    out = daily_monitor.expected_running_from_squeue(ssh_host="aircc", runner=runner)

    assert out == 3 * daily_monitor.N_SLOTS
    argv = seen["argv"]
    assert argv[0] == "ssh" and argv[3] == "aircc"
    assert len(argv) == 5 and argv[4].startswith("bash -lc ")   # login shell, one argv item
    assert "-n aircc_jm" in argv[4] and "-t RUNNING" in argv[4]


def test_expected_running_is_unknown_when_the_cluster_is_unreachable(tmp_path):
    def runner(argv):
        raise OSError("ssh: connect to host aircc port 22: Connection refused")

    assert daily_monitor.expected_running_from_squeue(runner=runner) is None


def test_expected_running_is_unknown_not_zero_when_squeue_is_empty(tmp_path):
    def runner(argv):
        return subprocess.CompletedProcess(argv, 0, stdout="\n", stderr="")

    # zero would make check_db assert "expected 0 running" and alert on a healthy
    # campaign the moment squeue hiccups
    assert daily_monitor.expected_running_from_squeue(runner=runner) is None


def test_check_db_skips_the_running_check_when_expected_is_unknown(tmp_path):
    out = daily_monitor.check_db(_db(tmp_path, running=31), expected=None)
    assert out.ok
    assert out.running_count == 31


def test_run_once_derives_the_expectation_and_does_not_cry_wolf(tmp_path):
    """A healthy campaign of any size must not alert -- the old fixed 32 did."""
    emails = []

    def runner(argv):
        assert "squeue" in argv[4]
        n = 44 // daily_monitor.N_SLOTS
        return subprocess.CompletedProcess(
            argv, 0, stdout="\n".join(f"153201_{i}" for i in range(n)), stderr="")

    rc = daily_monitor.run_once(
        db_path=_db(tmp_path, running=44),
        backup_log=_write_log(tmp_path),
        recommendations_dir=tmp_path / "rec",
        mount_repo=tmp_path / "mount",
        emailer=lambda s, b, **_: emails.append((s, b)),
        runner=runner,
    )

    assert rc == 0
    assert emails == []
