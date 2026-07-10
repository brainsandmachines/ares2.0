"""Deterministic failure classifier tests."""

from orchestrator.classify import (
    ACTION_FAIL,
    ACTION_REQUEUE,
    ACTION_UNKNOWN,
    classify,
    error_hash,
    extract_traceback,
)

TB = """some setup
Traceback (most recent call last):
  File "/repo/x.py", line 10, in run
    foo()
ValueError: bad shape
"""


def test_timeout_requeues():
    assert classify("slurmstepd: ... CANCELLED ... DUE TO TIME LIMIT ...") == ACTION_REQUEUE


def test_oom_requeues():
    assert classify("RuntimeError: CUDA out of memory. Tried to allocate 2GiB") == ACTION_REQUEUE


def test_empty_log_requeues():
    assert classify("") == ACTION_REQUEUE
    assert classify("   \n  ") == ACTION_REQUEUE


def test_no_traceback_requeues():
    assert classify("training epoch 5 ... node went away") == ACTION_REQUEUE


def test_known_code_errors_fail():
    assert classify("ModuleNotFoundError: No module named 'dvd'") == ACTION_FAIL
    assert classify(TB) == ACTION_FAIL                       # ValueError
    assert classify("FileNotFoundError: missing.yaml") == ACTION_FAIL


def test_unknown_traceback_is_unknown():
    log = ("Traceback (most recent call last):\n"
           "  File \"/x.py\", line 3, in run\n"
           "    weird()\n"
           "SomeExoticError: never seen this\n")
    assert classify(log) == ACTION_UNKNOWN


def test_extract_traceback():
    tb = extract_traceback(TB)
    assert tb.startswith("Traceback (most recent call last):")
    assert "ValueError: bad shape" in tb
    assert extract_traceback("no traceback here") == ""


def test_error_hash_stable_across_paths_and_numbers():
    a = 'File "/a/b.py", line 42, in run\nValueError: bad at 0xdead'
    b = 'File "/x/y.py", line 9, in run\nValueError: bad at 0xbeef'
    assert error_hash(a) == error_hash(b)
    assert error_hash(a) != error_hash("KeyError: nope")
