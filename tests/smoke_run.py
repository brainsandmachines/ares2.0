"""Smoke test: load the composed Hydra config without launching training."""
from omegaconf import OmegaConf

def main():
    cfg = OmegaConf.load('robust_training/configs/config.yaml')
    # The top-level config.yaml uses defaults to compose other groups when run by Hydra.
    # For smoke test, manually merge the specific files used in defaults.
    training = OmegaConf.load('robust_training/configs/training/convnext_small.yaml')
    model = OmegaConf.load('robust_training/configs/model/convnext_small.yaml')
    dataset = OmegaConf.load('robust_training/configs/dataset/imagenet.yaml')
    optimizer = OmegaConf.load('robust_training/configs/optimizer/adamw.yaml')
    attacks = OmegaConf.load('robust_training/configs/attacks/adv.yaml')
    dist = OmegaConf.load('robust_training/configs/dist/default.yaml')
    lr_scheduler = OmegaConf.load('robust_training/configs/lr_scheduler/cosine.yaml')
    continuation = OmegaConf.load('robust_training/configs/continuation/default.yaml')
    epsilon_schedule = OmegaConf.load('robust_training/configs/epsilon_schedule/default.yaml')
    checkpointing = OmegaConf.load('robust_training/configs/checkpointing/default.yaml')

    composed = OmegaConf.create({
        'training': training,
        'model': model,
        'dataset': dataset,
        'optimizer': optimizer,
        'attacks': attacks,
        'dist': dist,
        'lr_scheduler': lr_scheduler,
        'continuation': continuation,
        'epsilon_schedule': epsilon_schedule,
        'checkpointing': checkpointing,
    })

    # basic assertions
    assert composed.model.model
    assert composed.training.epochs
    assert composed.dataset.train_dir
    print('Smoke config OK. model=', composed.model.model, 'epochs=', composed.training.epochs)

if __name__ == '__main__':
    main()
