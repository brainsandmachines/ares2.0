import torch
from timm.data import Mixup, AugMixDataset, create_transform
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from torchvision import datasets

def build_dataset(cfg, num_aug_splits=0):
    '''The function to build dataset for robust training.'''
    dataset = cfg.dataset
    # build dataset
    dataset_train = datasets.ImageFolder(root=dataset.train_dir, transform=None)
    dataset_eval = datasets.ImageFolder(root=dataset.eval_dir, transform=None)
    # dataset_eval=ImageNet(root=dataset.eval_dir)

    # wrap dataset_train in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # build transform
    train_interpolation = dataset.train_interpolation
    if dataset.no_aug or not train_interpolation:
        train_interpolation = dataset.interpolation
    re_num_splits = 0
    if dataset.resplit:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset_train.transform = create_transform(
        dataset.input_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=dataset.no_aug,
        scale=dataset.scale,
        ratio=dataset.ratio,
        hflip=dataset.hflip,
        vflip=dataset.vflip,
        color_jitter=dataset.color_jitter,
        auto_augment=dataset.aa,
        interpolation=train_interpolation,
        mean=dataset.mean,
        std=dataset.std,
        crop_pct=dataset.crop_pct,
        tf_preprocessing=False,
        re_prob=dataset.reprob,
        re_mode=dataset.remode,
        re_count=dataset.recount,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0
    )

    dataset_eval.transform = create_transform(
        dataset.input_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=dataset.interpolation,
        mean=dataset.mean,
        std=dataset.std,
        crop_pct=dataset.crop_pct
    )

    # create sampler
    sampler_train = None
    sampler_eval = None
    if cfg.dist.distributed and not isinstance(dataset_train, torch.utils.data.IterableDataset):
        if dataset.aug_repeats:
            sampler_train = RepeatAugSampler(dataset_train, num_repeats=dataset.aug_repeats)
        else:
            sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        assert dataset.aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"
    sampler_eval = OrderedDistributedSampler(dataset_eval)
    
    # setup mixup / cutmix
    mixup_fn = None
    if dataset.mixup_active:
        mixup_args = dict(
            mixup_alpha=dataset.mixup, cutmix_alpha=dataset.cutmix, cutmix_minmax=dataset.cutmix_minmax,
            prob=dataset.mixup_prob, switch_prob=dataset.mixup_switch_prob, mode=dataset.mixup_mode,
            label_smoothing=dataset.smoothing, num_classes=cfg.model.num_classes)
        mixup_fn = Mixup(**mixup_args)
        
    collate_fn = mixup_collate_fn(mixup_fn) if mixup_fn is not None else None
    print(f"mixup_fn active: {mixup_fn is not None}")

    # create dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=cfg.training.batch_size,
        shuffle=not cfg.dist.distributed,
        num_workers=cfg.num_workers,
        sampler=sampler_train,
        collate_fn=collate_fn,
        pin_memory=cfg.pin_mem,
        drop_last=True
    )
    dataloader_eval = torch.utils.data.DataLoader(
        dataset=dataset_eval,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        sampler=sampler_eval,
        collate_fn=None,
        pin_memory=cfg.pin_mem,
        drop_last=False
    )

    return dataloader_train, dataloader_eval

def mixup_collate_fn(mixup_fn):
    def collate(batch):
        # batch = list of (image, label)
        images, labels = zip(*batch)          # unpack
        images = torch.stack(images, dim=0)   # (B, C, H, W)
        labels = torch.tensor(labels)         # (B)

        # apply mixup (timm Mixup returns mixed images + soft labels)
        images, labels = mixup_fn(images, labels)

        return images, labels
    return collate
