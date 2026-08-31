import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from aa_sweep import config, plan as plan_mod  # noqa: E402
from aa_sweep.tests.test_census import csv_text  # noqa: E402

ALL_CELLS = [(n, e) for n in config.NORMS for e in config.EPS_INPUTS]
CKPT = config.CKPT_FILE_FOR_KIND
BIG = 1_418_062_559


def make_db(path: Path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE jobs (model_name TEXT PRIMARY KEY, status TEXT)")
    conn.executemany("INSERT INTO jobs VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def probe(files: dict[str, int], csvs: dict[str, str] | None = None, exists=True):
    return {"exists": exists, "files": files, "csvs": csvs or {}}


def no_aircc(_name):
    return {}, {}


def no_botero(_name):
    """Keep the census hermetic: without this the default reader reads the real /mnt/data4t
    archive, and fixture model names that happen to exist there arrive with real CSV rows."""
    return {}, {}, None


def test_finished_models_selects_only_finished(tmp_path):
    db = tmp_path / "jobs.sqlite"
    make_db(db, [("a", "finished"), ("b", "running"), ("c", "finished"), ("d", "pending")])
    assert sorted(plan_mod.finished_models(db)) == ["a", "c"]


def test_finished_models_reports_a_missing_db_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="mount"):
        plan_mod.finished_models(tmp_path / "nope.sqlite")


def test_model_with_a_full_sweep_is_complete_and_stages_nothing():
    csvs = {k: csv_text("m", f"/x/m/{CKPT[k]}", ALL_CELLS) for k in config.CHECKPOINT_KINDS}
    works = plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({v: BIG for v in CKPT.values()}, csvs)},
        aircc_reader=no_aircc,
        botero_reader=no_botero,
    )
    assert works[0].is_complete
    assert works[0].runnable_kinds == []
    assert works[0].staging_files == []


def test_only_the_gapped_kinds_checkpoint_is_staged():
    """dvd_b_l2_1_init0's real shape: best+last swept, advbest never run."""
    csvs = {
        "best": csv_text("m", "results/models/m/model_best.pth.tar", ALL_CELLS),
        "last": csv_text("m", "results/models/m/last.pth.tar", ALL_CELLS),
    }

    def aircc_reader(_name):
        return {v: BIG for v in CKPT.values()}, csvs

    works = plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({}, {}, exists=False)},
        aircc_reader=aircc_reader,
        botero_reader=no_botero,
    )
    work = works[0]
    assert work.runnable_kinds == ["advbest"]
    assert work.staging_files == ["model_best_adv.pth.tar"]
    assert work.missing_cell_count == 15


def test_baseline_without_advbest_checkpoint_gets_no_advbest_job():
    def aircc_reader(_name):
        return {"model_best.pth.tar": BIG, "last.pth.tar": BIG}, {}

    works = plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({}, {}, exists=False)},
        aircc_reader=aircc_reader,
        botero_reader=no_botero,
    )
    assert works[0].runnable_kinds == ["best", "last"]
    assert "model_best_adv.pth.tar" not in works[0].staging_files


def test_differing_checkpoint_sizes_are_reported_as_a_conflict():
    """The 5 *pgd5* names exist as real BGU runs and as near-empty AIRCC husks."""
    def aircc_reader(_name):
        return {"last.pth.tar": 1}, {}

    works = plan_mod.build_plan(
        aircc_finished=["convnext_base_linf_cont4to6_pgd5_init0"], sjm_finished=[],
        slurm_probe={"convnext_base_linf_cont4to6_pgd5_init0": probe({"last.pth.tar": BIG})},
        aircc_reader=aircc_reader,
        botero_reader=no_botero,
    )
    work = works[0]
    assert any("last.pth.tar" in c and "keeping slurm" in c for c in work.conflicts)
    # The BGU checkpoint is already there, so nothing is copied over it.
    assert work.staging_files == []


def test_nested_sjm_name_resolves_dir_and_matches_rows_by_basename():
    name = "vit_b_cvst/linf_1_init1"
    csvs = {k: csv_text("linf_1_init1", f"/x/linf_1_init1/{CKPT[k]}", ALL_CELLS)
            for k in config.CHECKPOINT_KINDS}
    works = plan_mod.build_plan(
        aircc_finished=[], sjm_finished=[name],
        slurm_probe={name: probe({v: BIG for v in CKPT.values()}, csvs)},
        aircc_reader=no_aircc,
        botero_reader=no_botero,
    )
    work = works[0]
    assert work.slurm_dir.endswith("/results/models/vit_b_cvst/linf_1_init1")
    assert work.aircc_dir is None  # sjm-only: never staged
    assert work.is_complete


def test_sjm_model_is_never_staged_even_when_incomplete():
    works = plan_mod.build_plan(
        aircc_finished=[], sjm_finished=["vit_b_cvst/l2_1_init1"],
        slurm_probe={"vit_b_cvst/l2_1_init1": probe({v: BIG for v in CKPT.values()})},
        aircc_reader=no_aircc,
        botero_reader=no_botero,
    )
    work = works[0]
    assert work.runnable_kinds == ["best", "last", "advbest"]
    assert work.staging_files == []


def test_job_names_flatten_nested_model_names():
    assert config.job_name("vit_b_cvst/linf_1_init1", "best") == "aaswp_vit_b_cvst__linf_1_init1_best"
    assert config.job_name("convnext_base_l1_1_init0", "advbest") == "aaswp_convnext_base_l1_1_init0_advbest"
