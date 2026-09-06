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


def no_botero(_name):
    """Keep the census hermetic: without this the default reader reads the real local store, and
    fixture model names that happen to exist there arrive with real CSV rows."""
    return {}, {}, None


def botero_dir(files, csvs=None, path="/mnt/data4t/models/arch/m"):
    def reader(_name):
        return files, csvs or {}, Path(path)
    return reader


def test_finished_models_selects_only_finished(tmp_path):
    db = tmp_path / "jobs.sqlite"
    make_db(db, [("a", "finished"), ("b", "running"), ("c", "finished"), ("d", "pending")])
    assert sorted(plan_mod.finished_models(db)) == ["a", "c"]


def test_finished_models_reports_a_missing_db_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="mount"):
        plan_mod.finished_models(tmp_path / "nope.sqlite")


def test_a_model_on_the_cluster_belongs_to_the_slurm_lane():
    """Presence of a directory on the cluster decides the lane -- whichever DB finished it."""
    works = plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({v: BIG for v in CKPT.values()})},
        botero_reader=no_botero,
    )
    work = works[0]
    assert work.lane == config.SLURM_LANE
    assert work.slurm_dir.endswith("/results/models/m")
    assert work.botero_dir is None
    assert work.runnable_kinds == ["best", "last", "advbest"]


def test_an_aircc_model_absent_from_the_cluster_belongs_to_the_botero_lane():
    works = plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({}, {}, exists=False)},
        botero_reader=botero_dir({v: BIG for v in CKPT.values()}),
    )
    work = works[0]
    assert work.lane == config.BOTERO_LANE
    assert work.botero_dir == Path("/mnt/data4t/models/arch/m")
    assert work.runnable_kinds == ["best", "last", "advbest"]


def test_the_two_lanes_are_disjoint():
    """The property the per-lane census depends on: no model is ever in both lanes."""
    works = plan_mod.build_plan(
        aircc_finished=["on_cluster", "local_only"], sjm_finished=["sjm_only"],
        slurm_probe={
            "on_cluster": probe({v: BIG for v in CKPT.values()}),
            "sjm_only": probe({v: BIG for v in CKPT.values()}),
            "local_only": probe({}, {}, exists=False),
        },
        botero_reader=botero_dir({v: BIG for v in CKPT.values()}),
    )
    lanes = {w.model_name: w.lane for w in works}
    assert lanes == {
        "on_cluster": config.SLURM_LANE,
        "sjm_only": config.SLURM_LANE,
        "local_only": config.BOTERO_LANE,
    }


def test_an_aircc_model_with_no_local_copy_is_dropped_entirely():
    """The three `models_failed/` runs that never wrote a checkpoint: not the cluster's, not ours."""
    works = plan_mod.build_plan(
        aircc_finished=["convnext_base_linf_cont4to6_init0"], sjm_finished=[],
        slurm_probe={"convnext_base_linf_cont4to6_init0": probe({}, {}, exists=False)},
        botero_reader=no_botero,
    )
    assert works == []


def test_an_sjm_model_absent_from_the_cluster_is_dropped():
    """Not the cluster's to run (no dir) and not AIRCC's to hand us."""
    works = plan_mod.build_plan(
        aircc_finished=[], sjm_finished=["vit_b_cvst/l2_1_init1"],
        slurm_probe={"vit_b_cvst/l2_1_init1": probe({}, {}, exists=False)},
        botero_reader=no_botero,
    )
    assert works == []


def test_model_with_a_full_sweep_is_complete():
    csvs = {k: csv_text("m", f"/x/m/{CKPT[k]}", ALL_CELLS) for k in config.CHECKPOINT_KINDS}
    works = plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({v: BIG for v in CKPT.values()}, csvs)},
        botero_reader=no_botero,
    )
    assert works[0].is_complete
    assert works[0].runnable_kinds == []


def test_only_the_gapped_kind_is_runnable():
    """dvd_b_l2_1_init0's real shape: best+last swept, advbest never run."""
    csvs = {
        "best": csv_text("m", "results/models/m/model_best.pth.tar", ALL_CELLS),
        "last": csv_text("m", "results/models/m/last.pth.tar", ALL_CELLS),
    }
    works = plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({}, {}, exists=False)},
        botero_reader=botero_dir({v: BIG for v in CKPT.values()}, csvs),
    )
    work = works[0]
    assert work.runnable_kinds == ["advbest"]
    assert work.missing_cell_count == 15


def test_baseline_without_advbest_checkpoint_gets_no_advbest_job():
    works = plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({}, {}, exists=False)},
        botero_reader=botero_dir({"model_best.pth.tar": BIG, "last.pth.tar": BIG}),
    )
    assert works[0].runnable_kinds == ["best", "last"]


def test_nested_sjm_name_resolves_dir_and_matches_rows_by_basename():
    name = "vit_b_cvst/linf_1_init1"
    csvs = {k: csv_text("linf_1_init1", f"/x/linf_1_init1/{CKPT[k]}", ALL_CELLS)
            for k in config.CHECKPOINT_KINDS}
    works = plan_mod.build_plan(
        aircc_finished=[], sjm_finished=[name],
        slurm_probe={name: probe({v: BIG for v in CKPT.values()}, csvs)},
        botero_reader=no_botero,
    )
    work = works[0]
    assert work.slurm_dir.endswith("/results/models/vit_b_cvst/linf_1_init1")
    assert work.is_complete


def test_the_cluster_lane_never_reads_the_local_store():
    """A cell computed on Botero must not stop the cluster submitting it: the cluster's own engine
    diffs its own CSV, so a row it cannot see is a row it would recompute anyway."""
    calls = []

    def spy(name):
        calls.append(name)
        return {}, {}, None

    plan_mod.build_plan(
        aircc_finished=["m"], sjm_finished=[],
        slurm_probe={"m": probe({v: BIG for v in CKPT.values()})},
        botero_reader=spy,
    )
    assert calls == []


def test_job_names_flatten_nested_model_names():
    assert config.job_name("vit_b_cvst/linf_1_init1", "best") == "aaswp_vit_b_cvst__linf_1_init1_best"
    assert config.job_name("convnext_base_l1_1_init0", "advbest") == "aaswp_convnext_base_l1_1_init0_advbest"
