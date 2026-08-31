from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_mail_archive(tmp_path, monkeypatch):
    """Keep the tests out of the real ``logs/mail/`` archive.

    ``notify._spoolable`` archives every alert it routes, and archiving is on by
    default -- without this, running the suite would append test alerts to the
    live archive.
    """
    monkeypatch.setenv("ARES_MAIL_ARCHIVE", str(tmp_path / "mail_archive"))
