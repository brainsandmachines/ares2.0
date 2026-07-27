import sys
import types
import csv
import json
from pathlib import Path


class _DummyAutoAttack:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.modules.setdefault("autoattack", types.SimpleNamespace(AutoAttack=_DummyAutoAttack))

from data_analysis.autoattack_array_eval import (
    EPS_INPUTS,
    find_checkpoint_for_kind,
    is_complete_output,
    load_existing_selection,
    output_csv_for_checkpoint_kind,
    parse_checkpoint_kinds,
    select_balanced_indices,
    write_rows,
)


class _Dataset:
    def __init__(self, samples):
        self.samples = samples
        self.root = "/tmp/dataset"

    def __len__(self):
        return len(self.samples)


def test_select_balanced_indices_can_exceed_number_of_classes():
    samples = [(f"{class_idx}_{image_idx}.jpg", class_idx) for class_idx in range(1000) for image_idx in range(2)]
    ds = _Dataset(samples)

    selected = select_balanced_indices(ds, total_images=1500, seed=0)

    assert len(selected) == 1500
    assert len(set(selected)) == 1500


def test_select_balanced_indices_rejects_more_than_dataset_size():
    samples = [(f"{class_idx}_{image_idx}.jpg", class_idx) for class_idx in range(3) for image_idx in range(2)]
    ds = _Dataset(samples)

    try:
        select_balanced_indices(ds, total_images=7, seed=0)
    except ValueError as exc:
        assert "Need 7 images" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_default_eps_inputs_include_six_and_twelve():
    assert EPS_INPUTS == (1.0, 2.0, 4.0, 6.0, 8.0, 12.0)


def test_checkpoint_kind_csv_suffixes():
    assert output_csv_for_checkpoint_kind("autoattack_sweep_results.csv", "best") == "autoattack_sweep_results.csv"
    assert output_csv_for_checkpoint_kind("autoattack_sweep_results.csv", "last") == "autoattack_sweep_results_last.csv"
    assert output_csv_for_checkpoint_kind("autoattack_sweep_results.csv", "advbest") == "autoattack_sweep_results_advbest.csv"


def test_parse_checkpoint_kinds_rejects_unknown():
    assert parse_checkpoint_kinds("last,advbest") == ("last", "advbest")
    try:
        parse_checkpoint_kinds("last,nope")
    except ValueError as exc:
        assert "Unsupported checkpoint kind" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_find_checkpoint_for_kind_skips_missing_advbest(tmp_path):
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()
    (model_dir / "model_best.pth.tar").write_bytes(b"best")
    (model_dir / "last.pth.tar").write_bytes(b"last")

    assert find_checkpoint_for_kind(model_dir, "best") == model_dir / "model_best.pth.tar"
    assert find_checkpoint_for_kind(model_dir, "last") == model_dir / "last.pth.tar"
    assert find_checkpoint_for_kind(model_dir, "advbest") is None


def _aa_row(model_name, ckpt_path, norm, eps, run_id="old_run"):
    return {
        "run_id": run_id,
        "timestamp": "2026-06-16T00:00:00",
        "model_name": model_name,
        "checkpoint_path": str(ckpt_path),
        "state_dict_used": "state_dict_ema",
        "epoch": "199",
        "attack_norm": norm,
        "epsilon_input": eps,
        "epsilon_eval": eps,
        "clean_acc": "80.0",
        "robust_acc": "50.0",
        "num_images": "1024",
        "batch_size": "128",
        "num_batches": "8",
        "seed": "0",
        "selection_json": "autoattack_sweep_selection.json",
    }


def test_is_complete_output_accepts_mixed_run_ids(tmp_path):
    csv_path = tmp_path / "autoattack_sweep_results.csv"
    ckpt_path = tmp_path / "model_best.pth.tar"
    rows = []
    for norm in ("linf", "l2", "l1"):
        rows.append(_aa_row("model_a", ckpt_path, norm, 1.0, run_id="old_run"))
        rows.append(_aa_row("model_a", ckpt_path, norm, 6.0, run_id="backfill_run"))

    write_rows(csv_path, rows)

    assert is_complete_output(
        csv_path,
        max_settings=None,
        checkpoint_path=ckpt_path,
        model_name="model_a",
        run_id="new_default_run",
        eps_inputs=(1.0, 6.0),
    )


def test_is_complete_output_restricted_to_single_norm(tmp_path):
    csv_path = tmp_path / "autoattack_sweep_results.csv"
    ckpt_path = tmp_path / "model_best.pth.tar"
    # Only linf@8 is present; a full linf/l2/l1 sweep would be incomplete,
    # but restricting to norms=("linf",) with that single eps should be complete.
    write_rows(csv_path, [_aa_row("model_a", ckpt_path, "linf", 8.0)])

    assert is_complete_output(
        csv_path,
        max_settings=None,
        checkpoint_path=ckpt_path,
        model_name="model_a",
        eps_inputs=(8.0,),
        norms=("linf",),
    )
    assert not is_complete_output(
        csv_path,
        max_settings=None,
        checkpoint_path=ckpt_path,
        model_name="model_a",
        eps_inputs=(8.0,),
        norms=("linf", "l2", "l1"),
    )


def test_write_rows_upserts_only_requested_settings(tmp_path):
    csv_path = tmp_path / "autoattack_sweep_results.csv"
    ckpt_path = tmp_path / "model_best.pth.tar"
    write_rows(
        csv_path,
        [
            _aa_row("model_a", ckpt_path, "linf", 1.0),
            _aa_row("model_a", ckpt_path, "linf", 6.0, run_id="old_backfill"),
        ],
    )

    replacement = _aa_row("model_a", ckpt_path, "linf", 6.0, run_id="new_backfill")
    replacement["robust_acc"] = "42.0"
    write_rows(csv_path, [replacement], replace_settings={("linf", 6.0)})

    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert any(row["epsilon_input"] == "1.0" and row["run_id"] == "old_run" for row in rows)
    assert any(row["epsilon_input"] == "6.0" and row["run_id"] == "new_backfill" and row["robust_acc"] == "42.0" for row in rows)


def test_load_existing_selection_falls_back_to_model_local_basename(tmp_path):
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()
    csv_path = model_dir / "autoattack_sweep_results.csv"
    selection_path = model_dir / "autoattack_sweep_selection.json"
    selection_path.write_text(json.dumps({"selected_indices": [3, 1, 2]}))

    write_rows(
        csv_path,
        [
            {
                **_aa_row("model_a", model_dir / "model_best.pth.tar", "linf", 1.0),
                "selection_json": "results/models/model_a/autoattack_sweep_selection.json",
            }
        ],
    )

    loaded = load_existing_selection(model_dir, csv_path, "autoattack_sweep_selection.json")

    assert loaded is not None
    indices, path = loaded
    assert indices == [3, 1, 2]
    assert path == selection_path
