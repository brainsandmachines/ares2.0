import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aa_sweep import config  # noqa: E402
from aa_sweep.census import grid_cells, kind_status, observed_cells  # noqa: E402

HEADER = (
    "run_id,timestamp,model_name,checkpoint_kind,checkpoint_path,state_dict_used,epoch,"
    "attack_norm,epsilon_input,epsilon_eval,clean_acc,robust_acc,num_images,batch_size,"
    "num_batches,seed,selection_json\n"
)


def csv_text(model, ckpt_path, cells):
    rows = "".join(
        f"run,2026-01-01T00:00:00,{model},last,{ckpt_path},state_dict,40,"
        f"{norm},{eps},{eps},70.0,50.0,1024,128,8,0,sel.json\n"
        for norm, eps in cells
    )
    return HEADER + rows


GRID = grid_cells(config.NORMS, config.EPS_INPUTS)


def test_grid_is_15_cells_and_excludes_eps_12():
    assert len(GRID) == 15
    assert ("linf", 12.0) not in GRID
    assert ("l1", 8.0) in GRID


def test_observed_cells_matches_on_checkpoint_basename():
    """Rows written on AIRCC carry a relative path; they must still match after staging."""
    text = csv_text("m", "results/models/m/last.pth.tar", [("l2", 4.0)])
    assert observed_cells(text, "m", "last.pth.tar") == {("l2", 4.0)}


def test_observed_cells_ignores_other_checkpoints_and_models():
    text = (
        csv_text("m", "/x/m/model_best.pth.tar", [("l2", 1.0)])
        + csv_text("other", "/x/other/last.pth.tar", [("l2", 2.0)]).replace(HEADER, "")
    )
    assert observed_cells(text, "m", "last.pth.tar") == set()


def test_observed_cells_tolerates_missing_and_malformed_fields():
    text = HEADER + "run,,m,last,/x/m/last.pth.tar,sd,40,,,,,,,,,,\n"
    assert observed_cells(text, "m", "last.pth.tar") == set()
    assert observed_cells("", "m", "last.pth.tar") == set()


def test_kind_status_eps_norm_row_leaves_14_missing():
    status = kind_status(
        kind="last", ckpt_filename="last.pth.tar", model_name="m", grid=GRID,
        files={"last.pth.tar"},
        csv_text=csv_text("m", "results/models/m/last.pth.tar", [("l2", 4.0)]),
    )
    assert len(status.missing) == 14
    assert ("l2", 4.0) not in status.missing
    assert status.runnable


def test_kind_status_full_sweep_is_complete_even_with_a_stale_eps_12_row():
    cells = [(n, e) for n in config.NORMS for e in config.EPS_INPUTS] + [("linf", 12.0)]
    status = kind_status(
        kind="last", ckpt_filename="last.pth.tar", model_name="m", grid=GRID,
        files={"last.pth.tar"},
        csv_text=csv_text("m", "/x/m/last.pth.tar", cells),
    )
    assert status.missing == set()
    assert not status.runnable


def test_kind_status_reads_only_this_machines_csv():
    """One lane at a time: a row computed on the *other* machine is deliberately NOT counted.

    The engine on this machine diffs its own CSV against the grid, so a cell it cannot see is a
    cell it will recompute. Counting the other lane's row here would make the planner skip work
    that then never happens. Safe only because the lanes own disjoint model sets.
    """
    status = kind_status(
        kind="last", ckpt_filename="last.pth.tar", model_name="m", grid=GRID,
        files={"last.pth.tar"},
        csv_text=csv_text("m", "/x/m/last.pth.tar", [("l2", 1.0)]),
    )
    assert ("l2", 1.0) not in status.missing
    assert ("l2", 2.0) in status.missing        # only on the other machine -> still ours to do
    assert len(status.missing) == 14


def test_kind_status_without_a_checkpoint_here_is_not_runnable():
    """Baselines have no model_best_adv.pth.tar -- no job should ever be submitted for them.

    This is also what keeps a lane from being handed work it cannot do: `has_checkpoint` is about
    *this* machine's filesystem, so a model only the other machine holds is never runnable here.
    """
    status = kind_status(
        kind="advbest", ckpt_filename="model_best_adv.pth.tar", model_name="m", grid=GRID,
        files=set(), csv_text="",
    )
    assert status.missing == GRID
    assert not status.has_checkpoint and not status.runnable


def test_kind_status_is_runnable_with_a_local_checkpoint_and_gaps():
    status = kind_status(
        kind="advbest", ckpt_filename="model_best_adv.pth.tar", model_name="m", grid=GRID,
        files={"model_best_adv.pth.tar"}, csv_text="",
    )
    assert status.has_checkpoint and status.runnable and status.missing == GRID
