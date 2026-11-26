"""The `seed` CSV column: forwarded as a Hydra override, and sibling collisions refused."""

import csv
import sys

import pytest

from slurm_job_manager import seed as seed_cli
from slurm_job_manager.csv_spec import (
    build_overrides,
    load_all_rows,
    seed_collisions,
    sibling_key,
)


def test_seed_is_emitted_only_when_set():
    assert "seed=2" in build_overrides({"seed": "2", "model": "convnext_base"})
    assert not any(t.startswith("seed=") for t in build_overrides({"seed": "", "model": "convnext_base"}))
    # rows from CSVs without the column at all (every pre-2026-11 CSV)
    assert not any(t.startswith("seed=") for t in build_overrides({"model": "convnext_base"}))


def test_sibling_key_masks_only_the_init_token():
    assert sibling_key("convnext_base_l2_4_init1") == sibling_key("convnext_base_l2_4_init4")
    assert sibling_key("convnext_base_dvd_b_l1_cont4to8_init1_contepoch") == "convnext_base_dvd_b_l1_cont4to8_init*_contepoch"
    assert sibling_key("convnext_base_l2_4_init1") != sibling_key("convnext_base_l2_8_init1")


def _row(name, seed):
    return {"model_name": name, "seed": seed}


def test_seed_collisions():
    ok = [
        _row("a_l2_4_init0", "0"), _row("a_l2_4_init1", "1"),
        _row("a_l2_8_init0", "0"),            # same seed, different config: fine (seed = init)
        _row("a_linf_4_init1", ""),            # legacy empty seed: ignored
        _row("a_linf_4_init2", ""),
    ]
    assert seed_collisions(ok) == []

    bad = ok + [_row("a_l2_4_init2", "1")]
    (problem,) = seed_collisions(bad)
    assert "a_l2_4_init1" in problem and "a_l2_4_init2" in problem

    assert seed_collisions([_row("x_init1", "abc")])[0].endswith("is not an integer")
    assert "outside" in seed_collisions([_row("x_init1", str(2**32))])[0]


def test_live_csvs_have_no_seed_problems():
    assert seed_collisions(load_all_rows()) == []


def _write_csv(csv_dir, rows):
    cols = ["model_name", "arch", "init", "training.epochs", "priority", "is_test", "seed"]
    with (csv_dir / "convnext_base.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, init, seed in rows:
            w.writerow({"model_name": name, "arch": "convnext_base", "init": init,
                        "training.epochs": 10, "priority": 1, "is_test": "", "seed": seed})


def _run_seed(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(sys, "argv", ["seed", "--db", str(tmp_path / "jobs.sqlite"),
                                      "--arch", "convnext_base", "--csv-dir", str(tmp_path), *extra])
    seed_cli.main()


def test_seed_cli_refuses_collision_even_when_filtering_one_init(monkeypatch, tmp_path, capsys):
    _write_csv(tmp_path, [("m_l2_4_init1", 1, "1"), ("m_l2_4_init2", 2, "1")])
    with pytest.raises(SystemExit) as exc:
        _run_seed(monkeypatch, tmp_path, "--init", "2")
    assert "refusing" in str(exc.value)
    assert not (tmp_path / "jobs.sqlite").exists()  # refused before the DB was touched
    assert "shared by sibling rows" in capsys.readouterr().err


def test_seed_cli_accepts_distinct_seeds(monkeypatch, tmp_path, capsys):
    _write_csv(tmp_path, [("m_l2_4_init1", 1, "1"), ("m_l2_4_init2", 2, "2")])
    _run_seed(monkeypatch, tmp_path)
    assert "seeded 2 models" in capsys.readouterr().out
