"""Offline tests for deterministic failure classification."""

from __future__ import annotations

from slurm_job_manager.classify import (
    ACTION_FAIL,
    ACTION_REQUEUE,
    ACTION_UNKNOWN,
    classify,
    error_hash,
)

_TB = "Traceback (most recent call last):"


def test_time_limit_requeues():
    assert classify("slurmstepd: JOB 1 CANCELLED DUE TO TIME LIMIT") == ACTION_REQUEUE


def test_cuda_oom_requeues():
    assert classify("RuntimeError: CUDA out of memory. Tried to allocate...") == ACTION_REQUEUE


def test_empty_log_requeues():
    assert classify("") == ACTION_REQUEUE
    assert classify("   \n  ") == ACTION_REQUEUE


def test_deterministic_error_fails():
    log = f"{_TB}\n  File 'x.py', line 3\nValueError: bad config"
    assert classify(log) == ACTION_FAIL


def test_unknown_traceback_is_unknown():
    log = f"{_TB}\n  File 'x.py', line 3\nSomeWeirdError: mystery"
    assert classify(log) == ACTION_UNKNOWN


def test_clean_exit_without_traceback_requeues():
    assert classify("epoch 5 done\nprocess exited") == ACTION_REQUEUE


def test_error_hash_is_stable_across_paths_and_numbers():
    a = f"{_TB}\n  File '/a/b/x.py', line 3\nWeird: at 0xdeadbeef"
    b = f"{_TB}\n  File '/c/d/x.py', line 99\nWeird: at 0xcafef00d"
    assert error_hash(a) == error_hash(b)
