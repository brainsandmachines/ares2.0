import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aa_sweep import config, mirror  # noqa: E402

MTIME = 1_754_200_000


def _write(path: Path, size: int, mtime: int = MTIME):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))


def _make_pair(tmp_path, files, mirror_files=None):
    """Build an AIRCC-mount dir and a mirror dir. mirror_files defaults to an exact copy."""
    aircc = tmp_path / "aircc" / "m"
    mirror_dir = tmp_path / "mirror" / "m"
    for name, (size, mtime) in files.items():
        _write(aircc / name, size, mtime)
    for name, (size, mtime) in (files if mirror_files is None else mirror_files).items():
        _write(mirror_dir / name, size, mtime)
    return aircc, mirror_dir


def _point_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BACKUP_MIRROR", tmp_path / "mirror")
    monkeypatch.setattr(config, "USE_MIRROR", True)


# --- files_match -----------------------------------------------------------------------------

def test_files_match_accepts_an_exact_copy(tmp_path):
    aircc, mirror_dir = _make_pair(tmp_path, {"last.pth.tar": (100, MTIME)})
    ok, reason = mirror.files_match(mirror_dir, aircc, ["last.pth.tar"])
    assert ok, reason


def test_files_match_rejects_a_size_mismatch(tmp_path):
    """A half-copied checkpoint is the exact failure this guard exists for."""
    aircc, mirror_dir = _make_pair(
        tmp_path, {"last.pth.tar": (100, MTIME)}, {"last.pth.tar": (60, MTIME)}
    )
    ok, reason = mirror.files_match(mirror_dir, aircc, ["last.pth.tar"])
    assert not ok and "size" in reason


def test_files_match_rejects_a_stale_mtime(tmp_path):
    """Same size but an older mtime: the model was retrained since the last backup."""
    aircc, mirror_dir = _make_pair(
        tmp_path, {"last.pth.tar": (100, MTIME)}, {"last.pth.tar": (100, MTIME - 86400)}
    )
    ok, reason = mirror.files_match(mirror_dir, aircc, ["last.pth.tar"])
    assert not ok and "mtime" in reason


def test_files_match_rejects_a_file_missing_from_the_mirror(tmp_path):
    aircc, mirror_dir = _make_pair(tmp_path, {"last.pth.tar": (100, MTIME)}, {})
    ok, reason = mirror.files_match(mirror_dir, aircc, ["last.pth.tar"])
    assert not ok and "not in mirror" in reason


# --- choose_source ---------------------------------------------------------------------------

def test_verified_mirror_is_used(tmp_path, monkeypatch):
    aircc, mirror_dir = _make_pair(tmp_path, {
        "last.pth.tar": (100, MTIME),
        "autoattack_sweep_results_last.csv": (20, MTIME),
    })
    _point_config(monkeypatch, tmp_path)

    choice = mirror.choose_source("m", aircc, ["last.pth.tar"], backup_ok=True)

    assert choice.label == "mirror" and choice.path == mirror_dir


def test_a_failed_backup_forces_the_aircc_mount(tmp_path, monkeypatch):
    aircc, _ = _make_pair(tmp_path, {"last.pth.tar": (100, MTIME)})
    _point_config(monkeypatch, tmp_path)

    choice = mirror.choose_source("m", aircc, ["last.pth.tar"], backup_ok=False,
                                  backup_message="no done marker")

    assert choice.label == "aircc-mount" and choice.path == aircc
    assert "no done marker" in choice.reason


def test_model_missing_from_the_mirror_falls_back(tmp_path, monkeypatch):
    """A model that finished after the last 03:00 backup simply is not mirrored yet."""
    aircc, _ = _make_pair(tmp_path, {"last.pth.tar": (100, MTIME)}, {})
    _point_config(monkeypatch, tmp_path)

    choice = mirror.choose_source("brand_new_model", aircc, ["last.pth.tar"], backup_ok=True)

    assert choice.label == "aircc-mount"
    assert "not in mirror yet" in choice.reason


def test_stale_csv_in_the_mirror_forces_the_aircc_mount(tmp_path, monkeypatch):
    """The CSVs ride along with every stage; a stale one would hide cells that need running."""
    aircc, _ = _make_pair(
        tmp_path,
        {"last.pth.tar": (100, MTIME), "autoattack_sweep_results_last.csv": (900, MTIME)},
        {"last.pth.tar": (100, MTIME), "autoattack_sweep_results_last.csv": (20, MTIME - 3600)},
    )
    _point_config(monkeypatch, tmp_path)

    choice = mirror.choose_source("m", aircc, ["last.pth.tar"], backup_ok=True)

    assert choice.label == "aircc-mount"
    assert "autoattack_sweep_results_last.csv" in choice.reason


def test_mirror_can_be_disabled_by_config(tmp_path, monkeypatch):
    aircc, _ = _make_pair(tmp_path, {"last.pth.tar": (100, MTIME)})
    _point_config(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "USE_MIRROR", False)

    choice = mirror.choose_source("m", aircc, ["last.pth.tar"], backup_ok=True)

    assert choice.label == "aircc-mount" and choice.reason == "mirror disabled"


# --- backup_log_ok ---------------------------------------------------------------------------

GOOD_LOG = """[backup] 2026-08-03T03:00:01 rsync /src/ -> /dst/
Number of files: 1,234
Total file size: 100 bytes
sent 100 bytes  received 10 bytes
total size is 100  speedup is 1.00
[backup] 2026-08-03T03:20:00 done
"""


def test_backup_log_ok_accepts_a_clean_run(tmp_path):
    log = tmp_path / "backup.log"
    log.write_text(GOOD_LOG)
    ok, message = mirror.backup_log_ok(log)
    assert ok, message


def test_backup_log_ok_rejects_a_run_that_never_finished(tmp_path):
    log = tmp_path / "backup.log"
    log.write_text(GOOD_LOG.replace("[backup] 2026-08-03T03:20:00 done\n", ""))
    ok, _ = mirror.backup_log_ok(log)
    assert not ok


def test_backup_log_ok_rejects_a_missing_log(tmp_path):
    ok, message = mirror.backup_log_ok(tmp_path / "nope.log")
    assert not ok and "missing" in message


def test_backup_log_ok_judges_only_the_latest_run(tmp_path):
    """Yesterday succeeding does not license using a mirror today's run broke."""
    log = tmp_path / "backup.log"
    log.write_text(GOOD_LOG + "[backup] 2026-08-04T03:00:01 rsync /src/ -> /dst/\n"
                              "[backup] ERROR: rsync failed rc=12\n")
    ok, _ = mirror.backup_log_ok(log)
    assert not ok
