from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from research.v1_metric_pipeline.attacks.v1_penalty_attack import V1PenaltyAttack
from research.v1_metric_pipeline.metrics.v1_metric import V1PerceptualMetric
from research.v1_metric_pipeline.models.tiny_cnn import TinyCNN
from research.v1_metric_pipeline.models.v1_feature_extractor import V1FeatureExtractor
from research.v1_metric_pipeline.utils.config import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only adversarial training scaffold with V1 penalty attack")
    parser.add_argument("--config", type=str, default="research/v1_metric_pipeline/configs/train_scaffold.yaml")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    torch.manual_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    dataset = FakeData(
        size=cfg.train.dataset_size,
        image_size=(3, cfg.train.image_size_px, cfg.train.image_size_px),
        num_classes=cfg.train.num_classes,
        transform=ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    model = TinyCNN(in_channels=3, num_classes=cfg.train.num_classes).to(device)
    extractor = V1FeatureExtractor(cfg.extractor).to(device)
    metric = V1PerceptualMetric(extractor, cfg.metric).to(device)
    attack = V1PenaltyAttack(cfg.attack)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    model.train()
    for epoch in range(cfg.train.epochs):
        for step, (x, y) in enumerate(loader):
            if step >= cfg.train.max_batches_per_epoch:
                break
            x = x.to(device)
            y = y.to(device)

            x_adv = attack.generate(model=model, x=x, y=y, metric=metric)
            logits_adv = model(x_adv)
            loss = F.cross_entropy(logits_adv, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                clean_acc = (model(x).argmax(dim=1) == y).float().mean().item()
                adv_acc = (logits_adv.argmax(dim=1) == y).float().mean().item()
                v1_d = metric(x, x_adv, reduction="mean").item()

            print(
                f"epoch={epoch} step={step} loss={loss.item():.4f} "
                f"clean_acc={clean_acc:.3f} adv_acc={adv_acc:.3f} v1_d={v1_d:.5f}"
            )


if __name__ == "__main__":
    main()
