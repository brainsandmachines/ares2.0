"""Offline tests for the MD5-fenced failure analyzer."""

from orchestrator.db import OrchestratorDB
from orchestrator.failure_analyzer import (
    analyze_failure,
    extract_traceback,
    normalize_traceback,
    traceback_hash,
)

LOG_A = """2026-06-19 10:00:01 starting run
Traceback (most recent call last):
  File "/home/ashtomer/projects/ares/robust_training/x.py", line 42, in run
    foo(bar) at 0x7ffabc12
RuntimeError: CUDA out of memory (tried to allocate 0x55ee)
"""

# Same structural error: different timestamp, path, line number, addresses.
LOG_B = """2026-12-01 23:59:59 starting run
Traceback (most recent call last):
  File "/scratch/other/path/y.py", line 9999, in run
    foo(bar) at 0xdeadbeef
RuntimeError: CUDA out of memory (tried to allocate 0x1234)
"""

LOG_DIFFERENT = """Traceback (most recent call last):
  File "/x.py", line 1, in run
    g()
ValueError: bad shape
"""


def test_normalization_collapses_variable_tokens():
    assert normalize_traceback(LOG_A) == normalize_traceback(LOG_B)
    assert traceback_hash(LOG_A) == traceback_hash(LOG_B)


def test_distinct_errors_distinct_hash():
    assert traceback_hash(LOG_A) != traceback_hash(LOG_DIFFERENT)


def test_extract_traceback_prefers_last_block():
    tb = extract_traceback(LOG_A)
    assert tb.startswith("Traceback (most recent call last):")
    assert "starting run" not in tb


def test_extract_tail_fallback_when_no_traceback():
    tb = extract_traceback("line1\nline2\nline3", tail_lines=2)
    assert tb == "line2\nline3"


def test_analyze_dedup_new_then_seen(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("m1", "golan-trainmodels", "j1", "/d/m1", 0, 0, 250)
    db.upsert_model("m2", "golan-trainmodels", "j2", "/d/m2", 0, 0, 250)
    rec = str(tmp_path / "rec")

    llm_calls = []
    email_calls = []

    def llm(tb):
        llm_calls.append(tb)
        return "# fix\ndo the thing"

    def emailer(subj, body):
        email_calls.append(subj)

    o1 = analyze_failure(db, "m1", LOG_A, rec, llm_client=llm, emailer=emailer)
    o2 = analyze_failure(db, "m2", LOG_B, rec, llm_client=llm, emailer=emailer)

    assert o1.is_new and not o2.is_new
    assert o1.error_hash == o2.error_hash          # same signature
    assert len(llm_calls) == 1                     # escalated exactly once
    assert len(email_calls) == 1
    # both rows marked FAILED, with the signature recorded
    assert db.get_model_state("m1").status == "FAILED"
    assert db.get_model_state("m2").last_error_hash == o1.error_hash


def test_analyze_writes_report_without_llm(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("m1", "golan-trainmodels", "j1", "/d/m1", 0, 0, 250)
    rec = str(tmp_path / "rec")
    out = analyze_failure(db, "m1", LOG_DIFFERENT, rec)  # no llm/emailer
    assert out.is_new and out.report_path is not None
    with open(out.report_path) as fh:
        assert "Raw traceback" in fh.read()
