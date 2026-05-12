from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch
from omegaconf import OmegaConf
from timm.data import create_transform
from torchvision import datasets

from ares.model.v1_block import VOneBlock, build_v1_block


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VAL_DIR = Path("/mnt/data/datasets/imagenet/val")
DEFAULT_TRAIN_DIR = Path("/mnt/data/datasets/imagenet/train")
DEFAULT_MATRIX_PATH = Path("/mnt/data/tomerash/val_train_class_min_dists_l1_l2_linf_50000x1000x3.pt")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "v1_metric_pipeline" / "all" / "outputs" / "nearest_class_v1_study"
DEFAULT_MODEL_CFG_PATH = REPO_ROOT / "robust_training" / "configs" / "model" / "convnext_small_v1.yaml"
DEFAULT_DATASET_CFG_PATH = REPO_ROOT / "robust_training" / "configs" / "dataset" / "imagenet.yaml"
MATRIX_NORMS = ("l1", "l2", "linf")


@dataclass(frozen=True)
class StudyConfig:
    val_dir: str
    train_dir: str
    matrix_path: str
    output_dir: str
    model_cfg_path: str
    dataset_cfg_path: str
    num_val_samples: int
    num_train_per_pair: int
    seed: int
    device: str
    feature_batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pixel and post-V1 distances using a nearest-class matrix over ImageNet."
    )
    parser.add_argument("--val-dir", default=str(DEFAULT_VAL_DIR))
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--matrix-path", default=str(DEFAULT_MATRIX_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-cfg-path", default=str(DEFAULT_MODEL_CFG_PATH))
    parser.add_argument("--dataset-cfg-path", default=str(DEFAULT_DATASET_CFG_PATH))
    parser.add_argument("--num-val-samples", type=int, default=1000)
    parser.add_argument("--num-train-per-pair", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    return parser.parse_args()


def load_cfg(path: Path) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def build_eval_dataset(root: str, dataset_cfg: Dict) -> datasets.ImageFolder:
    transform = create_transform(
        dataset_cfg["input_size"],
        is_training=False,
        use_prefetcher=False,
        interpolation=dataset_cfg["interpolation"],
        mean=dataset_cfg["mean"],
        std=dataset_cfg["std"],
        crop_pct=dataset_cfg["crop_pct"],
    )
    return datasets.ImageFolder(root=root, transform=transform)


def build_clean_v1_block(model_cfg: Dict, dataset_cfg: Dict, device: torch.device) -> VOneBlock:
    v1_block = build_v1_block(
        image_size=dataset_cfg["input_size"],
        visual_degrees=model_cfg["v1_visual_degrees"],
        stride=model_cfg["v1_stride"],
        ksize=model_cfg["v1_ksize"],
        sf_corr=model_cfg["v1_sf_corr"],
        sf_max=model_cfg["v1_sf_max"],
        sf_min=model_cfg["v1_sf_min"],
        rand_param=model_cfg["v1_rand_param"],
        gabor_seed=model_cfg["v1_gabor_seed"],
        simple_channels=model_cfg["v1_simple_channels"],
        complex_channels=model_cfg["v1_complex_channels"],
        noise_mode=None,
        noise_scale=model_cfg["v1_noise_scale"],
        noise_level=model_cfg["v1_noise_level"],
        k_exc=model_cfg["v1_k_exc"],
    )
    return v1_block.to(device).eval()


def make_run_dir(output_dir: str) -> Path:
    run_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sample_indices(dataset_size: int, count: int, seed: int) -> List[int]:
    if count > dataset_size:
        raise ValueError(f"Requested {count} samples from dataset of size {dataset_size}")
    rng = random.Random(seed)
    return rng.sample(list(range(dataset_size)), count)


def load_dataset_examples(
    dataset: datasets.ImageFolder,
    indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    tensors: List[torch.Tensor] = []
    labels: List[int] = []
    paths: List[str] = []
    for idx in indices:
        image, label = dataset[idx]
        tensors.append(image)
        labels.append(label)
        paths.append(dataset.samples[idx][0])
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long), paths


def denormalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
    return images * std_t + mean_t


def compute_pair_distances(anchor: torch.Tensor, others: torch.Tensor) -> Dict[str, torch.Tensor]:
    diffs = (others - anchor.unsqueeze(0)).reshape(others.shape[0], -1)
    return {
        "l1": diffs.abs().sum(dim=1),
        "l2": diffs.norm(p=2, dim=1),
        "linf": diffs.abs().max(dim=1).values,
    }


def summarize_frame(df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False)
    records = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for value_col in value_cols:
            row[f"{value_col}_mean"] = float(group[value_col].mean())
            row[f"{value_col}_std"] = float(group[value_col].std(ddof=0))
            row[f"{value_col}_min"] = float(group[value_col].min())
            row[f"{value_col}_max"] = float(group[value_col].max())
        records.append(row)
    return pd.DataFrame(records)


def build_class_to_indices(dataset: datasets.ImageFolder) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(int(label), []).append(idx)
    return class_to_indices


def select_nearest_class(distances: torch.Tensor, forbidden_class: int) -> tuple[int, float]:
    if distances.ndim != 1:
        raise ValueError(f"Expected 1D class-distance tensor, got shape {tuple(distances.shape)}")
    allowed = distances.clone()
    if not 0 <= forbidden_class < allowed.shape[0]:
        raise ValueError(f"Forbidden class {forbidden_class} is outside class range {allowed.shape[0]}")
    allowed[forbidden_class] = 0.0
    candidate_mask = allowed > 0
    if not torch.any(candidate_mask):
        raise ValueError("No strictly positive distances remain after excluding the source class")
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    best_local_index = int(torch.argmin(allowed[candidate_indices]).item())
    best_class = int(candidate_indices[best_local_index].item())
    return best_class, float(allowed[best_class].item())


def extract_v1_features(
    v1_block: VOneBlock,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        with torch.no_grad():
            feats = v1_block(batch, add_noise=False).detach().cpu()
        outputs.append(feats)
    return torch.cat(outputs, dim=0)


def sample_train_indices_for_class(
    class_to_indices: Dict[int, List[int]],
    class_id: int,
    count: int,
    rng: random.Random,
) -> List[int]:
    candidates = class_to_indices.get(class_id, [])
    if len(candidates) < count:
        raise ValueError(f"Class {class_id} has {len(candidates)} samples, cannot draw {count}")
    return rng.sample(candidates, count)


def validate_matrix_shape(matrix: torch.Tensor, val_len: int, num_classes: int) -> None:
    expected_shape = (val_len, num_classes, len(MATRIX_NORMS))
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Expected matrix shape {expected_shape}, got {tuple(matrix.shape)}")


def run_study(config: StudyConfig) -> Dict[str, object]:
    device = torch.device(config.device)
    run_dir = make_run_dir(config.output_dir)

    model_cfg = load_cfg(Path(config.model_cfg_path))
    dataset_cfg = load_cfg(Path(config.dataset_cfg_path))
    val_dataset = build_eval_dataset(config.val_dir, dataset_cfg)
    train_dataset = build_eval_dataset(config.train_dir, dataset_cfg)
    class_to_indices = build_class_to_indices(train_dataset)

    matrix = torch.load(config.matrix_path, map_location="cpu")
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Expected matrix tensor, got {type(matrix)}")
    validate_matrix_shape(matrix, len(val_dataset), len(train_dataset.classes))

    v1_block = build_clean_v1_block(model_cfg, dataset_cfg, device)
    val_indices = sample_indices(len(val_dataset), config.num_val_samples, config.seed)
    val_tensors, val_labels, val_paths = load_dataset_examples(val_dataset, val_indices)
    val_pixels = denormalize(val_tensors, dataset_cfg["mean"], dataset_cfg["std"])
    val_features = extract_v1_features(v1_block, val_tensors, device, config.feature_batch_size)

    train_rng = random.Random(config.seed + 1)
    rows: List[Dict[str, object]] = []

    for sample_pos, val_index in enumerate(val_indices):
        val_label = int(val_labels[sample_pos].item())
        val_pixel = val_pixels[sample_pos]
        val_feature = val_features[sample_pos]
        class_distance_row = matrix[val_index]

        for norm_index, selector_norm in enumerate(MATRIX_NORMS):
            nearest_class, matrix_distance = select_nearest_class(class_distance_row[:, norm_index], val_label)
            sampled_train_indices = sample_train_indices_for_class(
                class_to_indices,
                nearest_class,
                config.num_train_per_pair,
                train_rng,
            )

            train_tensors, train_labels, train_paths = load_dataset_examples(train_dataset, sampled_train_indices)
            train_pixels = denormalize(train_tensors, dataset_cfg["mean"], dataset_cfg["std"])
            train_features = extract_v1_features(v1_block, train_tensors, device, config.feature_batch_size)
            pixel_distances = compute_pair_distances(val_pixel, train_pixels)
            v1_distances = compute_pair_distances(val_feature, train_features)

            for local_idx, train_index in enumerate(sampled_train_indices):
                row = {
                    "val_index": val_index,
                    "val_path": val_paths[sample_pos],
                    "val_label": val_label,
                    "selector_norm": selector_norm,
                    "nearest_class_index": nearest_class,
                    "matrix_nearest_distance": matrix_distance,
                    "train_index": train_index,
                    "train_path": train_paths[local_idx],
                    "train_label": int(train_labels[local_idx].item()),
                    "pixel_l1": float(pixel_distances["l1"][local_idx].item()),
                    "pixel_l2": float(pixel_distances["l2"][local_idx].item()),
                    "pixel_linf": float(pixel_distances["linf"][local_idx].item()),
                    "v1_l1": float(v1_distances["l1"][local_idx].item()),
                    "v1_l2": float(v1_distances["l2"][local_idx].item()),
                    "v1_linf": float(v1_distances["linf"][local_idx].item()),
                }
                rows.append(row)

    raw_df = pd.DataFrame(rows)
    summary_values = ["pixel_l1", "pixel_l2", "pixel_linf", "v1_l1", "v1_l2", "v1_linf"]
    summary_by_selector_norm = summarize_frame(raw_df, ["selector_norm"], summary_values)
    summary_by_selected_class = summarize_frame(raw_df, ["selector_norm", "nearest_class_index"], summary_values)

    raw_df.to_csv(run_dir / "raw_pairs.csv", index=False)
    summary_by_selector_norm.to_csv(run_dir / "summary_by_selector_norm.csv", index=False)
    summary_by_selected_class.to_csv(run_dir / "summary_by_selected_class.csv", index=False)

    metadata = {
        "val_dir": config.val_dir,
        "train_dir": config.train_dir,
        "matrix_path": config.matrix_path,
        "seed": config.seed,
        "num_val_samples": config.num_val_samples,
        "num_train_per_pair": config.num_train_per_pair,
        "device": str(device),
        "feature_batch_size": config.feature_batch_size,
        "model_cfg_path": config.model_cfg_path,
        "dataset_cfg_path": config.dataset_cfg_path,
        "matrix_norm_order": list(MATRIX_NORMS),
        "val_indices": val_indices,
        "pixel_distance_space": "denormalized_[0,1]",
        "feature_distance_space": "post_v1_block_flattened",
        "same_class_excluded": True,
        "v1_noise_enabled": False,
    }
    with (run_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "run_dir": run_dir,
        "raw_pairs": raw_df,
        "summary_by_selector_norm": summary_by_selector_norm,
        "summary_by_selected_class": summary_by_selected_class,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    result = run_study(
        StudyConfig(
            val_dir=args.val_dir,
            train_dir=args.train_dir,
            matrix_path=args.matrix_path,
            output_dir=args.output_dir,
            model_cfg_path=args.model_cfg_path,
            dataset_cfg_path=args.dataset_cfg_path,
            num_val_samples=args.num_val_samples,
            num_train_per_pair=args.num_train_per_pair,
            seed=args.seed,
            device=args.device,
            feature_batch_size=args.feature_batch_size,
        )
    )
    print(f"Saved outputs to: {result['run_dir']}")


if __name__ == "__main__":
    main()

