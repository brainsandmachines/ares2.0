import argparse

import torch
from omegaconf import OmegaConf

import ares.utils.dataset as dataset_mod


def _build_args():
    cfg = OmegaConf.load("robust_training/configs/dataset/imagenet.yaml")
    args = argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))
    args.batch_size = 4
    args.num_workers = 2
    args.distributed = False
    args.pin_mem = True
    args.num_classes = 1000
    return args


def test_build_dataset_preserves_transform_and_mixup_contract(monkeypatch):
    args = _build_args()
    transform_calls = []
    mixup_calls = []
    dataloader_calls = []

    class _FakeImageFolder:
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform

        def __len__(self):
            return 8

        def __getitem__(self, idx):
            image = torch.zeros(3, 8, 8)
            label = idx % 2
            if callable(self.transform):
                image = self.transform(image)
            return image, label

    def _fake_create_transform(input_size, is_training, **kwargs):
        transform_calls.append({"input_size": input_size, "is_training": is_training, **kwargs})

        def _transform(x):
            return x

        return _transform

    class _FakeMixup:
        def __init__(self, **kwargs):
            mixup_calls.append(kwargs)

        def __call__(self, images, labels):
            soft = torch.nn.functional.one_hot(labels, num_classes=args.num_classes).float()
            return images + 1.0, soft

    class _FakeLoader:
        def __init__(self, **kwargs):
            dataloader_calls.append(kwargs)
            self.dataset = kwargs["dataset"]
            self.batch_size = kwargs["batch_size"]
            self.shuffle = kwargs["shuffle"]
            self.num_workers = kwargs["num_workers"]
            self.sampler = kwargs.get("sampler")
            self.collate_fn = kwargs.get("collate_fn")
            self.pin_memory = kwargs["pin_memory"]
            self.drop_last = kwargs["drop_last"]

        def __iter__(self):
            raise AssertionError("Fake loader should not be iterated in this unit test")

    monkeypatch.setattr(dataset_mod.datasets, "ImageFolder", _FakeImageFolder)
    monkeypatch.setattr(dataset_mod, "create_transform", _fake_create_transform)
    monkeypatch.setattr(dataset_mod, "Mixup", _FakeMixup)
    monkeypatch.setattr(dataset_mod, "OrderedDistributedSampler", lambda dataset: ("ordered", dataset))
    monkeypatch.setattr(dataset_mod.torch.utils.data, "DataLoader", _FakeLoader)

    loader_train, loader_eval = dataset_mod.build_dataset(args, num_aug_splits=0)

    assert len(transform_calls) == 2
    train_transform = transform_calls[0]
    eval_transform = transform_calls[1]

    assert train_transform["is_training"] is True
    assert train_transform["auto_augment"] == args.aa
    assert train_transform["re_prob"] == args.reprob
    assert train_transform["re_mode"] == args.remode
    assert train_transform["re_count"] == args.recount
    assert train_transform["interpolation"] == args.train_interpolation

    assert eval_transform["is_training"] is False
    assert eval_transform["interpolation"] == args.interpolation
    assert eval_transform["crop_pct"] == args.crop_pct

    assert len(mixup_calls) == 1
    mixup_kwargs = mixup_calls[0]
    assert mixup_kwargs["mixup_alpha"] == args.mixup
    assert mixup_kwargs["cutmix_alpha"] == args.cutmix
    assert mixup_kwargs["prob"] == args.mixup_prob
    assert mixup_kwargs["switch_prob"] == args.mixup_switch_prob
    assert mixup_kwargs["mode"] == args.mixup_mode
    assert mixup_kwargs["label_smoothing"] == args.smoothing

    train_loader_kwargs = dataloader_calls[0]
    eval_loader_kwargs = dataloader_calls[1]
    assert train_loader_kwargs["drop_last"] is True
    assert eval_loader_kwargs["drop_last"] is False
    assert loader_train.collate_fn is not None
    assert loader_eval.collate_fn is None

    batch = [(torch.zeros(3, 8, 8), 1), (torch.ones(3, 8, 8), 2)]
    mixed_images, mixed_labels = loader_train.collate_fn(batch)
    assert mixed_images.shape == (2, 3, 8, 8)
    assert mixed_labels.shape == (2, args.num_classes)
    assert torch.allclose(mixed_labels.sum(dim=1), torch.ones(2))


def test_build_dataset_without_mixup_preserves_hard_labels(monkeypatch):
    args = _build_args()
    args.mixup_active = False

    class _FakeImageFolder:
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform

        def __len__(self):
            return 4

    class _FakeLoader:
        def __init__(self, **kwargs):
            self.collate_fn = kwargs.get("collate_fn")

    monkeypatch.setattr(dataset_mod.datasets, "ImageFolder", _FakeImageFolder)
    monkeypatch.setattr(dataset_mod, "create_transform", lambda *a, **k: (lambda x: x))
    monkeypatch.setattr(dataset_mod, "OrderedDistributedSampler", lambda dataset: ("ordered", dataset))
    monkeypatch.setattr(dataset_mod.torch.utils.data, "DataLoader", _FakeLoader)

    loader_train, _loader_eval = dataset_mod.build_dataset(args, num_aug_splits=0)

    assert loader_train.collate_fn is None
