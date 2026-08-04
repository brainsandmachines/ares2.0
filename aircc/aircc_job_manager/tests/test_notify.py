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
