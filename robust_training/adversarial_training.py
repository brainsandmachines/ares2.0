import warnings
warnings.filterwarnings("ignore")
import argparse
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import yaml
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.tensorboard import SummaryWriter

# timm functions
from timm.models import resume_checkpoint, load_checkpoint, model_parameters
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import ModelEmaV2, distribute_bn, reduce_tensor, dispatch_clip_grad, get_outdir, CheckpointSaver, update_summary

# robust training functions
from ares.utils.dist import distributed_init, random_seed
from ares.utils.logger import setup_logger , _auto_experiment_name
from ares.utils.model import build_model
from ares.utils.loss import build_loss, resolve_amp, build_loss_scaler
from ares.utils.dataset import build_dataset
from ares.utils.adv import adv_generator
from ares.utils.metrics import AverageMeter, accuracy
from ares.utils.defaults import get_args_parser
from ares.utils.train_loop import train_one_epoch
from ares.utils.validate import validate

from model_scheduler import Model_scheduler

def main(args):
    # distributed settings and logger
    # if "WORLD_SIZE" in os.environ:
    #     args.world_size=int(os.environ["WORLD_SIZE"])
    args.distributed=float(args.world_size)>1
    distributed_init(args)
    # normalize attack eps/step for linf (values historically stored as 0-255)
    if getattr(args, 'attack_norm', None) == 'linf':
        if getattr(args, 'attack_eps', None) is not None:
            args.attack_eps = float(args.attack_eps) / 255.0
        if getattr(args, 'attack_step', None) is not None:
            args.attack_step = float(args.attack_step) / 255.0
    _logger = setup_logger(save_dir=None, distributed_rank=args.rank)
    _logger.info(f"Runtime distributed={args.distributed}, world_size={args.world_size}, rank={args.rank}, local_rank={args.local_rank}, device_id={args.device_id}")

    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark = True
    
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # resolve amp
    resolve_amp(args, _logger)

    # build model
    model = build_model(args, _logger, num_aug_splits)

    # create optimizer
    optimizer=None
    if args.lr is None:
        args.lr=args.lrb * args.batch_size * args.world_size / 512
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # build loss scaler
    amp_autocast, loss_scaler = build_loss_scaler(args, _logger)

    # resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            log_info=args.rank == 0)

    # setup ema
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # # setup distributed training
    if args.distributed:
        _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[args.device_id])
        # NOTE: EMA model does not need to be wrapped by DDP
    
    # create the train and eval dataloaders
    loader_train, loader_eval, mixup_fn = build_dataset(args, num_aug_splits)

    # setup loss function
    train_loss_fn, validate_loss_fn = build_loss(args, mixup_fn, num_aug_splits)

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)
    _logger.info('Scheduled epochs: {}'.format(num_epochs))
    
    sch = Model_scheduler(db_path="model_scheduler.db")

    args.output_dir = os.path.join(args.output_dir or ".", args.experiment_name)

    if args.rank == 0:
        _logger.info(f"Experiment: {args.experiment_name}")
        _logger.info(f"Results directory: {args.output_dir}")
    # saver
    eval_metric = args.eval_metric
    saver = None
    best_metric = None
    best_epoch = None
    output_dir = None
    writer = None
    if args.rank == 0:
        output_dir = get_outdir(args.output_dir)
        _logger = setup_logger(save_dir=output_dir, distributed_rank=args.rank)
        _logger.info(f"Experiment directory: {output_dir}")
        decreasing=True if eval_metric=='loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.max_history)
        TB_BASE = "/home/ashtomer/projects/ares/robust_training/tensorboard_logs"
        if "adv=0" in args.model_id:
            group_name = "baseline"
        else:
            group_name = args.attack_norm
        tb_dir = os.path.join(TB_BASE, group_name, args.experiment_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        
        # Add metadata for clarity
        writer.add_text("config/model_id", args.model_id)
        writer.add_text("config/attack_norm", str(args.attack_norm))
        writer.add_text("config/constraint", str(args.attack_eps))
        writer.add_text("config/seed", str(args.model.experiment_num))
        
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # start training
    _logger.info(f"Start training for {args.epochs} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        if hasattr(loader_train, 'sampler') and hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)
        # one epoch training
        train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, amp_autocast=amp_autocast,
                loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn, _logger=_logger, writer=writer,
                sch=sch, model_id=args.model_id)

        # distributed bn sync
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        # calculate evaluation metric
        eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, _logger=_logger, writer=writer, epoch=epoch, tb_tag='val')

        # model ema update
        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
            ema_eval_metrics = validate(model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)', _logger=_logger, writer=writer, epoch=epoch, tb_tag='val_ema')
            eval_metrics = ema_eval_metrics

        # lr_scheduler update
        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

        # output summary.csv
        if output_dir is not None:
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

        if saver is not None:
            save_path = os.path.join(saver.checkpoint_dir, f"checkpoint-{epoch}.pth.tar")
            # If a checkpoint with this name already exists, remove it to avoid FileExistsError
            if os.path.exists(save_path):
                if args.rank == 0:
                    print(f"[Warning] Removing existing checkpoint: {save_path}")
                    os.remove(save_path)
                # Ensure all ranks see the same file state before continuing
                if args.distributed:
                    torch.distributed.barrier()

            best_metric, best_epoch = saver.save_checkpoint(epoch, eval_metrics[eval_metric])

        if writer is not None and args.rank == 0:
            writer.flush()

        if args.distributed:
            torch.distributed.barrier()

    if writer is not None and args.rank == 0:
        writer.flush()
        writer.close()
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))



def _cfg_to_namespace(cfg):
    """Convert a DictConfig (Hydra/OmegaConf) or dict to argparse.Namespace expected by main().

    The original code expected a flat Namespace. Our Hydra configs are grouped (training, model,
    dataset, optimizer, attacks). Merge known groups into a single flat dict and return Namespace.
    """
    if isinstance(cfg, argparse.Namespace):
        return cfg
    if isinstance(cfg, DictConfig) or isinstance(cfg, dict):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        merged = {}
        # take top-level scalars
        for k, v in cfg_dict.items():
            if not isinstance(v, dict):
                merged[k] = v
        # merge known groups
        for group in ('training', 'model', 'dataset', 'optimizer', 'attacks'):
            if group in cfg_dict and cfg_dict[group] is not None:
                grp = cfg_dict[group]
                if isinstance(grp, dict):
                    merged.update(grp)
        return argparse.Namespace(**merged)
    # fallback
    return argparse.Namespace(**dict(cfg))


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def hydra_main(cfg: DictConfig):
    """Hydra entrypoint: composes configs and calls existing main()."""
    args = _cfg_to_namespace(cfg)
    if getattr(args, 'attack_norm', None) == 'linf':
        if getattr(args, 'attack_eps', None) is not None:
            args.attack_eps = args.attack_eps / 255.0
        if getattr(args, 'attack_step', None) is not None:
            args.attack_step = args.attack_step / 255.0
    main(args)


if __name__ == '__main__':
    # keep the original argparse CLI for backward compatibility
    parser = argparse.ArgumentParser('Robust training script', parents=[get_args_parser()])
    args = parser.parse_args()
    opt = vars(args)
    if args.configs:
        opt.update(yaml.load(open(args.configs), Loader=yaml.FullLoader))
    
    args = argparse.Namespace(**opt)
    main(args)
