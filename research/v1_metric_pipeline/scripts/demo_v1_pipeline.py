from __future__ import annotations

import argparse
from pathlib import Path

import torch

from research.v1_metric_pipeline.attacks.v1_penalty_attack import V1PenaltyAttack
from research.v1_metric_pipeline.metrics.v1_metric import V1PerceptualMetric
from research.v1_metric_pipeline.models.tiny_cnn import TinyCNN
from research.v1_metric_pipeline.models.v1_feature_extractor import V1FeatureExtractor
from research.v1_metric_pipeline.utils.config import load_pipeline_config
from research.v1_metric_pipeline.utils.io import load_image_as_tensor, save_tensor_image


def run_demo(config_path: str, image_a: str, image_b: str, output_dir: str | None = None) -> dict:
    cfg = load_pipeline_config(config_path)
    torch.manual_seed(cfg.demo.seed)

    device = torch.device(cfg.demo.device)
    out_dir = Path(output_dir or cfg.demo.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xa = load_image_as_tensor(image_a, cfg.extractor.image_size_px).to(device)
    xb = load_image_as_tensor(image_b, cfg.extractor.image_size_px).to(device)

    extractor = V1FeatureExtractor(cfg.extractor).to(device)
    metric = V1PerceptualMetric(extractor, cfg.metric).to(device)
    attack = V1PenaltyAttack(cfg.attack)
    model = TinyCNN(in_channels=3, num_classes=cfg.demo.model_num_classes).to(device)
    model.eval()

    with torch.no_grad():
        d_ab = metric(xa, xb, reduction="mean").item()
        y = model(xa).argmax(dim=1)

    x_adv = attack.generate(model=model, x=xa, y=y, metric=metric)
    with torch.no_grad():
        d_adv = metric(xa, x_adv, reduction="mean").item()
        logits_clean = model(xa)
        logits_adv = model(x_adv)
        pred_clean = int(logits_clean.argmax(dim=1).item())
        pred_adv = int(logits_adv.argmax(dim=1).item())

    save_tensor_image(xa[0], str(out_dir / "image_a.png"))
    save_tensor_image(xb[0], str(out_dir / "image_b.png"))
    save_tensor_image(x_adv[0], str(out_dir / "image_a_adv.png"))
    save_tensor_image((x_adv - xa).abs()[0], str(out_dir / "delta_abs.png"))

    result = {
        "v1_distance_image_a_vs_b": d_ab,
        "v1_distance_image_a_vs_adv": d_adv,
        "pred_clean": pred_clean,
        "pred_adv": pred_adv,
        "output_dir": str(out_dir),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: deterministic V1 metric + V1-penalty attack")
    parser.add_argument("--config", type=str, default="research/v1_metric_pipeline/configs/demo.yaml")
    parser.add_argument("--image-a", type=str, required=True)
    parser.add_argument("--image-b", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_demo(args.config, args.image_a, args.image_b, args.output_dir)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
