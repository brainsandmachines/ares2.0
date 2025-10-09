"""Smoke test: load the composed Hydra config (without launching training)
and verify conversion to argparse.Namespace works. This is fast and safe.
"""
from omegaconf import OmegaConf
from robust_training.adversarial_training import _cfg_to_namespace

def main():
    cfg = OmegaConf.load('robust_training/configs/config.yaml')
    # The top-level config.yaml uses defaults to compose other groups when run by Hydra.
    # For smoke test, manually merge the specific files used in defaults.
    training = OmegaConf.load('robust_training/configs/training/convnext_small.yaml')
    model = OmegaConf.load('robust_training/configs/model/convnext_small.yaml')
    dataset = OmegaConf.load('robust_training/configs/dataset/imagenet.yaml')
    optimizer = OmegaConf.load('robust_training/configs/optimizer/adamw.yaml')
    attacks = OmegaConf.load('robust_training/configs/attacks/adv.yaml')

    composed = OmegaConf.create({
        'training': training,
        'model': model,
        'dataset': dataset,
        'optimizer': optimizer,
        'attacks': attacks,
    })

    args = _cfg_to_namespace(composed)
    # basic assertions
    assert hasattr(args, 'model')
    assert hasattr(args, 'epochs')
    assert hasattr(args, 'train_dir')
    print('Smoke conversion OK. model=', args.model, 'epochs=', args.epochs)

if __name__ == '__main__':
    main()
