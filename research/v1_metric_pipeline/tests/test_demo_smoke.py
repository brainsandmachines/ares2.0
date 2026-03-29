from pathlib import Path

import torch
from torchvision.utils import save_image

from research.v1_metric_pipeline.scripts.demo_v1_pipeline import run_demo


def test_demo_smoke(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    out_dir = tmp_path / "out"

    save_image(torch.rand(3, 96, 96), str(image_a))
    save_image(torch.rand(3, 96, 96), str(image_b))

    result = run_demo(
        config_path="research/v1_metric_pipeline/configs/demo.yaml",
        image_a=str(image_a),
        image_b=str(image_b),
        output_dir=str(out_dir),
    )

    assert "v1_distance_image_a_vs_b" in result
    assert "v1_distance_image_a_vs_adv" in result
    assert (out_dir / "image_a.png").exists()
    assert (out_dir / "image_a_adv.png").exists()
