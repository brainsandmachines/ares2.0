from __future__ import annotations

import argparse
from dataclasses import asdict

import yaml

from research.v1_metric_pipeline.utils.config import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Print resolved V1 pipeline hyperparameters")
    parser.add_argument("--config", type=str, default="research/v1_metric_pipeline/configs/v1_defaults.yaml")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    print(yaml.safe_dump(asdict(cfg), sort_keys=False))


if __name__ == "__main__":
    main()
