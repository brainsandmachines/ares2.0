from __future__ import annotations

import argparse

from research.v1_metric_pipeline.models.v1_feature_extractor import V1FeatureExtractor
from research.v1_metric_pipeline.utils.config import load_pipeline_config
from research.v1_metric_pipeline.utils.io import load_image_as_tensor
from research.v1_metric_pipeline.utils.visualization import save_feature_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize simple/complex V1 response maps for an image")
    parser.add_argument("--config", type=str, default="research/v1_metric_pipeline/configs/v1_defaults.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out-simple", type=str, default="research/v1_metric_pipeline/outputs/simple_maps.png")
    parser.add_argument("--out-complex", type=str, default="research/v1_metric_pipeline/outputs/complex_maps.png")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    x = load_image_as_tensor(args.image, cfg.extractor.image_size_px)
    extractor = V1FeatureExtractor(cfg.extractor)
    feats = extractor(x)
    save_feature_grid(feats["simple"], args.out_simple)
    save_feature_grid(feats["complex"], args.out_complex)
    print(f"Saved simple maps to: {args.out_simple}")
    print(f"Saved complex maps to: {args.out_complex}")


if __name__ == "__main__":
    main()
