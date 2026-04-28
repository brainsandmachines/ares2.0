from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf
from timm.data import create_transform
from torchvision import datasets

from ares.model.v1_convnext import V1ConvNeXt
from ares.utils.adv import adv_generator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VAL_DIR = Path("/mnt/data/datasets/imagenet_sample/val")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "v1_metric_pipeline" / "outputs" / "v1_eps_scaling"
MODEL_CFG_PATH = REPO_ROOT / "robust_training" / "configs" / "model" / "convnext_small_v1.yaml"
DATASET_CFG_PATH = REPO_ROOT / "robust_training" / "configs" / "dataset" / "imagenet.yaml"
ATTACK_CFG_PATH = REPO_ROOT / "robust_training" / "configs" / "attacks" / "adv.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect how pixel-space eps maps into post-V1 feature distances.")
    parser.add_argument("--val-dir", type=str, default=str(DEFAULT_VAL_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--pgd-batch-size", type=int, default=16)
    parser.add_argument("--eps-list", type=str, default="1,2,4,8,16")
    parser.add_argument("--norms", type=str, default="linf,l2,l1")
    parser.add_argument("--attack-steps", type=int, default=3)
    parser.add_argument("--num-noise-samples", type=int, default=5)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_cfg(path: Path) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def build_eval_dataset(val_dir: str, dataset_cfg: Dict) -> datasets.ImageFolder:
    transform = create_transform(
        dataset_cfg["input_size"],
        is_training=False,
        use_prefetcher=False,
        interpolation=dataset_cfg["interpolation"],
        mean=dataset_cfg["mean"],
        std=dataset_cfg["std"],
        crop_pct=dataset_cfg["crop_pct"],
    )
    return datasets.ImageFolder(root=val_dir, transform=transform)


def build_v1_model(model_cfg: Dict, dataset_cfg: Dict, checkpoint: str | None, device: torch.device) -> V1ConvNeXt:
    model = V1ConvNeXt(
        backbone_name="convnext_small",
        input_size=dataset_cfg["input_size"],
        v1_noise_train_only=model_cfg.get("v1_noise_train_only", True),
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
        noise_mode=model_cfg["v1_noise_mode"],
        noise_scale=model_cfg["v1_noise_scale"],
        noise_level=model_cfg["v1_noise_level"],
        k_exc=model_cfg["v1_k_exc"],
    ).to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def make_run_dir(output_dir: str) -> Path:
    run_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_csv_arg(values: str, cast=float) -> List:
    return [cast(v.strip()) for v in values.split(",") if v.strip()]


def sample_indices(dataset_size: int, count: int, seed: int) -> List[int]:
    if count > dataset_size:
        raise ValueError(f"Requested {count} samples from dataset of size {dataset_size}")
    rng = random.Random(seed)
    return rng.sample(list(range(dataset_size)), count)


def load_dataset_examples(dataset: datasets.ImageFolder, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
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


def pair_distance_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
    flat = (a - b).view(a.shape[0], -1)
    return {
        "l1": flat.abs().sum(dim=1),
        "l2": flat.norm(p=2, dim=1),
        "linf": flat.abs().max(dim=1).values,
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


def extract_v1_features(
    model: V1ConvNeXt,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
    apply_noise: bool,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        with torch.no_grad():
            feats = model.forward_v1_features(batch, apply_noise=apply_noise).detach().cpu()
        outputs.append(feats)
    return torch.cat(outputs, dim=0)


def compute_v1_distance_records(
    model: V1ConvNeXt,
    a: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_noise_samples: int,
) -> Dict[str, torch.Tensor]:
    clean_a = extract_v1_features(model, a, device, batch_size, apply_noise=False)
    clean_b = extract_v1_features(model, b, device, batch_size, apply_noise=False)
    clean = pair_distance_tensors(clean_a, clean_b)

    noisy_values = {norm: [] for norm in ("l1", "l2", "linf")}
    for _ in range(num_noise_samples):
        noisy_a = extract_v1_features(model, a, device, batch_size, apply_noise=True)
        noisy_b = extract_v1_features(model, b, device, batch_size, apply_noise=True)
        noisy = pair_distance_tensors(noisy_a, noisy_b)
        for norm, values in noisy.items():
            noisy_values[norm].append(values)

    result: Dict[str, torch.Tensor] = {}
    for norm in ("l1", "l2", "linf"):
        stacked = torch.stack(noisy_values[norm], dim=0)
        result[f"v1_clean_{norm}"] = clean[norm]
        result[f"v1_noisy_{norm}_mean"] = stacked.mean(dim=0)
        result[f"v1_noisy_{norm}_std"] = stacked.std(dim=0, unbiased=False)
    return result


def run_pair_study(
    dataset: datasets.ImageFolder,
    model: V1ConvNeXt,
    dataset_cfg: Dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, Dict]:
    pair_indices = sample_indices(len(dataset), args.num_pairs * 2, args.seed)
    tensors, _, paths = load_dataset_examples(dataset, pair_indices)
    pixels = denormalize(tensors, dataset_cfg["mean"], dataset_cfg["std"])

    a = tensors[0::2]
    b = tensors[1::2]
    pixel_a = pixels[0::2]
    pixel_b = pixels[1::2]
    path_a = paths[0::2]
    path_b = paths[1::2]

    pixel_dist = pair_distance_tensors(pixel_a, pixel_b)
    v1_dist = compute_v1_distance_records(model, a, b, device, args.feature_batch_size, args.num_noise_samples)

    rows = []
    for pair_id in range(args.num_pairs):
        row = {
            "pair_id": pair_id,
            "image_a_index": pair_indices[2 * pair_id],
            "image_b_index": pair_indices[2 * pair_id + 1],
            "image_a_path": path_a[pair_id],
            "image_b_path": path_b[pair_id],
        }
        for norm in ("l1", "l2", "linf"):
            row[f"pixel_{norm}"] = float(pixel_dist[norm][pair_id])
            row[f"v1_clean_{norm}"] = float(v1_dist[f"v1_clean_{norm}"][pair_id])
            row[f"v1_noisy_{norm}_mean"] = float(v1_dist[f"v1_noisy_{norm}_mean"][pair_id])
            row[f"v1_noisy_{norm}_std"] = float(v1_dist[f"v1_noisy_{norm}_std"][pair_id])
        rows.append(row)

    metadata = {
        "pair_seed": args.seed,
        "pair_indices": pair_indices,
        "pair_paths": paths,
        "pixel_distance_space": "denormalized_[0,1]",
        "feature_distance_space": "post_v1_block",
    }
    return pd.DataFrame(rows), metadata


def prepare_attack_hparams(norm: str, configured_eps: float, attack_steps: int) -> tuple[float, float]:
    effective_eps = float(configured_eps)
    effective_step = None
    if norm == "linf":
        effective_eps = effective_eps / 255.0
        effective_step = effective_eps / max(int(attack_steps), 1)
    elif norm == "l1":
        effective_eps = effective_eps * 255.0 / 2.0
        effective_step = 1.0
    elif norm == "l2":
        effective_step = 2.0 * effective_eps / max(int(attack_steps), 1)
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    return effective_eps, effective_step


def build_attack_args(dataset_cfg: Dict, attack_cfg: Dict, norm: str, device: torch.device) -> SimpleNamespace:
    amp_version = "native" if device.type == "cuda" else "none"
    return SimpleNamespace(
        std=dataset_cfg["std"],
        mean=dataset_cfg["mean"],
        attack_norm=norm,
        amp_version=amp_version,
        l1_step_mode=attack_cfg["l1_step_mode"],
        l1_apgd_rho=attack_cfg["l1_apgd_rho"],
        l1_apgd_use_halving=attack_cfg["l1_apgd_use_halving"],
        l1_apgd_min_step_scale=attack_cfg["l1_apgd_min_step_scale"],
    )


def run_pgd_study(
    dataset: datasets.ImageFolder,
    model: V1ConvNeXt,
    dataset_cfg: Dict,
    attack_cfg: Dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, Dict]:
    pgd_indices = sample_indices(len(dataset), args.pgd_batch_size, args.seed + 1)
    clean_images, labels, paths = load_dataset_examples(dataset, pgd_indices)
    clean_pixels = denormalize(clean_images, dataset_cfg["mean"], dataset_cfg["std"])

    records = []
    metadata_conditions = []
    norms = parse_csv_arg(args.norms, cast=str)
    eps_values = parse_csv_arg(args.eps_list, cast=float)

    for norm in norms:
        for configured_eps in eps_values:
            effective_eps, effective_step = prepare_attack_hparams(norm, configured_eps, args.attack_steps)
            attack_args = build_attack_args(dataset_cfg, attack_cfg, norm, device)
            model.train()
            adv_images = adv_generator(
                attack_args,
                clean_images.to(device),
                labels.to(device),
                model,
                effective_eps,
                args.attack_steps,
                effective_step,
                random_start=False,
                use_best=True,
            ).detach().cpu()
            model.eval()

            adv_pixels = denormalize(adv_images, dataset_cfg["mean"], dataset_cfg["std"])
            pixel_dist = pair_distance_tensors(clean_pixels, adv_pixels)
            v1_dist = compute_v1_distance_records(
                model,
                clean_images,
                adv_images,
                device,
                args.feature_batch_size,
                args.num_noise_samples,
            )

            for idx in range(clean_images.shape[0]):
                row = {
                    "image_index": pgd_indices[idx],
                    "image_path": paths[idx],
                    "attack_norm": norm,
                    "configured_eps": configured_eps,
                    "effective_eps": effective_eps,
                    "effective_step": effective_step,
                    "attack_steps": args.attack_steps,
                }
                for distance_norm in ("l1", "l2", "linf"):
                    row[f"pixel_{distance_norm}"] = float(pixel_dist[distance_norm][idx])
                    row[f"v1_clean_{distance_norm}"] = float(v1_dist[f"v1_clean_{distance_norm}"][idx])
                    row[f"v1_noisy_{distance_norm}_mean"] = float(v1_dist[f"v1_noisy_{distance_norm}_mean"][idx])
                    row[f"v1_noisy_{distance_norm}_std"] = float(v1_dist[f"v1_noisy_{distance_norm}_std"][idx])
                records.append(row)

            metadata_conditions.append(
                {
                    "attack_norm": norm,
                    "configured_eps": configured_eps,
                    "effective_eps": effective_eps,
                    "effective_step": effective_step,
                    "attack_steps": args.attack_steps,
                    "random_start": False,
                    "use_best": True,
                }
            )

    metadata = {
        "pgd_seed": args.seed + 1,
        "pgd_indices": pgd_indices,
        "pgd_paths": paths,
        "conditions": metadata_conditions,
        "pixel_distance_space": "denormalized_[0,1]",
        "feature_distance_space": "post_v1_block",
    }
    return pd.DataFrame(records), metadata


def plot_pair_histograms(pair_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, norm in zip(axes, ("l1", "l2", "linf")):
        ax.hist(pair_df[f"pixel_{norm}"], bins=20, alpha=0.5, label="pixel")
        ax.hist(pair_df[f"v1_clean_{norm}"], bins=20, alpha=0.5, label="v1_clean")
        ax.hist(pair_df[f"v1_noisy_{norm}_mean"], bins=20, alpha=0.5, label="v1_noisy_mean")
        ax.set_title(norm.upper())
        ax.set_xlabel("distance")
        ax.set_ylabel("count")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pgd_curves(pgd_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=False)
    norm_to_metric = {"linf": "linf", "l2": "l2", "l1": "l1"}

    for ax, attack_norm in zip(axes, ("linf", "l2", "l1")):
        subset = pgd_summary[pgd_summary["attack_norm"] == attack_norm].sort_values("configured_eps")
        metric = norm_to_metric[attack_norm]
        ax.plot(subset["configured_eps"], subset[f"pixel_{metric}_mean"], marker="o", label=f"pixel_{metric}")
        ax.plot(subset["configured_eps"], subset[f"v1_clean_{metric}_mean"], marker="o", label=f"v1_clean_{metric}")
        ax.plot(
            subset["configured_eps"],
            subset[f"v1_noisy_{metric}_mean_mean"],
            marker="o",
            label=f"v1_noisy_{metric}",
        )
        ax.set_title(f"attack={attack_norm}")
        ax.set_xlabel("configured eps")
        ax.set_ylabel("mean distance")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_metadata(path: Path, payload: Dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = make_run_dir(args.output_dir)

    model_cfg = load_cfg(MODEL_CFG_PATH)
    dataset_cfg = load_cfg(DATASET_CFG_PATH)
    attack_cfg = load_cfg(ATTACK_CFG_PATH)

    dataset = build_eval_dataset(args.val_dir, dataset_cfg)
    model = build_v1_model(model_cfg, dataset_cfg, args.checkpoint, device)

    pair_df, pair_meta = run_pair_study(dataset, model, dataset_cfg, args, device)
    pair_summary = summarize_frame(
        pair_df.assign(kind="pairs"),
        group_cols=["kind"],
        value_cols=[
            "pixel_l1",
            "pixel_l2",
            "pixel_linf",
            "v1_clean_l1",
            "v1_clean_l2",
            "v1_clean_linf",
            "v1_noisy_l1_mean",
            "v1_noisy_l2_mean",
            "v1_noisy_linf_mean",
        ],
    )

    pgd_df, pgd_meta = run_pgd_study(dataset, model, dataset_cfg, attack_cfg, args, device)
    pgd_summary = summarize_frame(
        pgd_df,
        group_cols=["attack_norm", "configured_eps", "effective_eps", "effective_step", "attack_steps"],
        value_cols=[
            "pixel_l1",
            "pixel_l2",
            "pixel_linf",
            "v1_clean_l1",
            "v1_clean_l2",
            "v1_clean_linf",
            "v1_noisy_l1_mean",
            "v1_noisy_l2_mean",
            "v1_noisy_linf_mean",
        ],
    )

    pair_df.to_csv(run_dir / "pair_distances.csv", index=False)
    pair_summary.to_csv(run_dir / "pair_distance_summary.csv", index=False)
    pgd_df.to_csv(run_dir / "pgd_v1_distances.csv", index=False)
    pgd_summary.to_csv(run_dir / "pgd_v1_distance_summary.csv", index=False)

    plot_pair_histograms(pair_df, run_dir / "pair_distance_histograms.png")
    plot_pgd_curves(pgd_summary, run_dir / "pgd_distance_curves.png")

    metadata = {
        "val_dir": args.val_dir,
        "device": str(device),
        "seed": args.seed,
        "num_pairs": args.num_pairs,
        "pgd_batch_size": args.pgd_batch_size,
        "eps_list": parse_csv_arg(args.eps_list, float),
        "norms": parse_csv_arg(args.norms, str),
        "attack_steps": args.attack_steps,
        "num_noise_samples": args.num_noise_samples,
        "feature_batch_size": args.feature_batch_size,
        "checkpoint": args.checkpoint,
        "model_cfg": model_cfg,
        "dataset_cfg": dataset_cfg,
        "attack_cfg": attack_cfg,
        "pair_study": pair_meta,
        "pgd_study": pgd_meta,
    }
    write_metadata(run_dir / "metadata.json", metadata)

    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
