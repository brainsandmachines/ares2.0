from __future__ import annotations

import argparse

from research.v1_metric_pipeline.models.v1_feature_extractor import V1FeatureExtractor
from research.v1_metric_pipeline.utils.config import load_pipeline_config
from research.v1_metric_pipeline.utils.visualization import save_kernel_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a grid visualization of V1 Gabor kernels")
    parser.add_argument("--config", type=str, default="research/v1_metric_pipeline/configs/v1_defaults.yaml")
    parser.add_argument("--out", type=str, default="research/v1_metric_pipeline/outputs/gabor_grid.png")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    extractor = V1FeatureExtractor(cfg.extractor)
    kernels = extractor.get_gabor_kernels()
    save_kernel_grid(kernels, args.out)
    print(f"Saved Gabor grid to: {args.out}")


if __name__ == "__main__":
    main()
