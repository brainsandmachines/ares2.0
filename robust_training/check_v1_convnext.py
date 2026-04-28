import argparse

import torch

from ares.model.v1_convnext import V1ConvNeXt


def main():
    parser = argparse.ArgumentParser(description="Sanity check for V1 + ConvNeXt-S forward pass")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--noise-mode", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = V1ConvNeXt(
        backbone_name="convnext_small",
        input_size=args.input_size,
        noise_mode=args.noise_mode,
    ).to(device)
    model.eval()

    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    with torch.no_grad():
        v1_features = model.forward_v1_features(x)
        logits = model(x)

    print(f"input_shape={tuple(x.shape)}")
    print(f"v1_shape={tuple(v1_features.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
