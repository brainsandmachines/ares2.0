from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image

from research.v1_metric_pipeline.all.run_nearest_class_v1_study import MATRIX_NORMS, StudyConfig, run_study, select_nearest_class


def _write_imagefolder_dataset(root: Path, images_per_class: dict[str, int], image_size: int = 64) -> None:
    for class_name, count in images_per_class.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            value = ((hash((class_name, idx)) % 200) + 20) / 255.0
            image = torch.full((3, image_size, image_size), fill_value=value, dtype=torch.float32)
            save_image(image, str(class_dir / f"{idx:04d}.png"))


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_select_nearest_class_excludes_source_and_zero_values() -> None:
    distances = torch.tensor([0.0, 1.25, 0.5, 0.0], dtype=torch.float32)
    chosen_class, chosen_distance = select_nearest_class(distances, forbidden_class=2)

    assert MATRIX_NORMS == ("l1", "l2", "linf")
    assert chosen_class == 1
    assert chosen_distance == 1.25


def test_run_study_smoke(tmp_path: Path) -> None:
    val_dir = tmp_path / "val"
    train_dir = tmp_path / "train"
    output_dir = tmp_path / "out"
    model_cfg_path = tmp_path / "model_cfg.yaml"
    dataset_cfg_path = tmp_path / "dataset_cfg.yaml"
    matrix_path = tmp_path / "matrix.pt"

    _write_imagefolder_dataset(val_dir, {"class0": 1, "class1": 1, "class2": 1})
    _write_imagefolder_dataset(train_dir, {"class0": 3, "class1": 3, "class2": 3})

    matrix = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.5, 2.0], [2.0, 1.0, 1.5]],
            [[3.0, 2.0, 0.5], [0.0, 0.0, 0.0], [1.0, 3.0, 2.0]],
            [[2.5, 1.0, 1.2], [1.0, 2.0, 0.8], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    torch.save(matrix, matrix_path)

    _write_yaml(
        model_cfg_path,
        {
            "v1_visual_degrees": 8,
            "v1_stride": 2,
            "v1_ksize": 15,
            "v1_sf_corr": 0.75,
            "v1_sf_max": 9,
            "v1_sf_min": 0,
            "v1_rand_param": False,
            "v1_gabor_seed": 0,
            "v1_simple_channels": 8,
            "v1_complex_channels": 8,
            "v1_noise_scale": 0.35,
            "v1_noise_level": 0.07,
            "v1_k_exc": 25,
        },
    )
    _write_yaml(
        dataset_cfg_path,
        {
            "input_size": 64,
            "interpolation": "bicubic",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 1.0,
        },
    )

    result = run_study(
        StudyConfig(
            val_dir=str(val_dir),
            train_dir=str(train_dir),
            matrix_path=str(matrix_path),
            output_dir=str(output_dir),
            model_cfg_path=str(model_cfg_path),
            dataset_cfg_path=str(dataset_cfg_path),
            num_val_samples=2,
            num_train_per_pair=2,
            seed=0,
            device="cpu",
            feature_batch_size=2,
        )
    )

    raw_pairs = result["raw_pairs"]
    run_dir = Path(result["run_dir"])

    assert len(raw_pairs) == 2 * 3 * 2
    assert set(raw_pairs["selector_norm"]) == set(MATRIX_NORMS)
    assert raw_pairs["pixel_l1"].ge(0).all()
    assert raw_pairs["v1_l2"].ge(0).all()
    assert (run_dir / "raw_pairs.csv").exists()
    assert (run_dir / "summary_by_selector_norm.csv").exists()
    assert (run_dir / "summary_by_selected_class.csv").exists()
    assert (run_dir / "metadata.json").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["v1_noise_enabled"] is False
    assert metadata["pixel_distance_space"] == "denormalized_[0,1]"
