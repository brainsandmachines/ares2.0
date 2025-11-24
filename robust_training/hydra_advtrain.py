"""Hydra entrypoint that composes config groups and calls the existing training main.

This loader merges the group configs (training, model, dataset, optimizer, attacks)
into a flat argparse.Namespace so the existing `adversarial_training.main(args)` can be
re-used without refactoring the training code.

Usage examples:
  python robust_training/hydra_advtrain.py  # uses defaults from configs/
  python robust_training/hydra_advtrain.py training.epochs=200 optimizer.weight_decay=0.1

"""
from __future__ import annotations
import argparse
import os
from typing import Dict

from omegaconf import OmegaConf
import hydra


def _merge_groups(cfg) -> Dict:
    # cfg will contain group nodes like cfg.training, cfg.model, etc.
    merged: Dict = {}
    # First take any top-level scalar keys
    for k, v in cfg.items():
        if not isinstance(v, dict) and not hasattr(v, '_get_node'):  # primitive
            merged[k] = v
    # Merge known groups if present
    for group in ('training', 'model', 'dataset', 'optimizer', 'attacks', 'dist','loss', 'lr_scheduler'):
        if group in cfg:
            group_dict = OmegaConf.to_container(cfg[group], resolve=True)
            if group_dict:
                merged.update(group_dict)
    return merged


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def hydra_main(cfg) -> None:
    # Merge group configs into a flat dict for argparse compatibility
    merged = _merge_groups(cfg)

    # Resolve any OmegaConf containers
    merged = OmegaConf.create(merged)
    merged = OmegaConf.to_container(merged, resolve=True)

    # Convert to argparse.Namespace (the training script expects that)
    args = argparse.Namespace(**merged)

    # attack eps/step normalization is handled inside adversarial_training.main()

    # Import here to avoid heavy imports when parsing config only
    from robust_training.adversarial_training import main as training_main

    training_main(args)


if __name__ == '__main__':
    hydra_main()
