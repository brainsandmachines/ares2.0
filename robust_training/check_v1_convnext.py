import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ares.model.v1_convnext import V1ConvNeXt


@hydra.main(config_path=None, config_name=None, version_base="1.3")
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(
        {
            "batch_size": 2,
            "input_size": 224,
            "device": "cpu",
            "noise_mode": None,
        },
        cfg,
    )

    device = torch.device(cfg.device)
    model = V1ConvNeXt(
        backbone_name="convnext_small",
        input_size=cfg.input_size,
        noise_mode=cfg.noise_mode,
    ).to(device)
    model.eval()

    x = torch.randn(cfg.batch_size, 3, cfg.input_size, cfg.input_size, device=device)

    with torch.no_grad():
        v1_features = model.forward_v1_features(x)
        logits = model(x)

    print(f"input_shape={tuple(x.shape)}")
    print(f"v1_shape={tuple(v1_features.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
