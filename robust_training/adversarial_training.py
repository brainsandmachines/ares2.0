import warnings
warnings.filterwarnings("ignore")
import argparse
from omegaconf import DictConfig, OmegaConf
import hydra
import yaml
import os
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import wandb

# timm functions
from timm.models import resume_checkpoint, load_checkpoint
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import ModelEmaV2, distribute_bn, get_outdir, CheckpointSaver, update_summary

# robust training functions
from ares.utils.dist import distributed_init, random_seed
from ares.utils.logger import setup_logger , _auto_experiment_name
from ares.utils.model import build_model
from ares.utils.loss import build_loss, resolve_amp, build_loss_scaler
from ares.utils.gradnorm import DBP
from ares.utils.dataset import build_dataset
from ares.utils.defaults import get_args_parser
from ares.utils.train_loop import train_one_epoch
from ares.utils.validate import validate

from job_manager.model_scheduler import Model_scheduler

from omegaconf import DictConfig, OmegaConf
import hydra


def main(args):
    # distributed settings and logger
    if "WORLD_SIZE" in os.environ:
        args.world_size=int(os.environ["WORLD_SIZE"])
    args.distributed=float(args.world_size)>1
    distributed_init(args)
    
    # normalize attack eps/step for linf (values historically stored as 0-255)
    if getattr(args, 'attack_norm', None) == 'linf':
        if getattr(args, 'attack_eps', None) is not None:
            args.attack_eps = float(args.attack_eps) / 255.0
        if getattr(args, 'attack_step', None) is not None:
            args.attack_step = float(args.attack_step) / 255.0
    if args.rank == 0:
        experiment_name, group_name = _auto_experiment_name(args)
        args.output_dir = os.path.join(args.output_dir,experiment_name)
        os.makedirs(args.output_dir, exist_ok = True)
    
    _logger = setup_logger(save_dir=args.output_dir, distributed_rank=args.rank)
    _logger.info(f"Runtime distributed={args.distributed}, world_size={args.world_size}, rank={args.rank}, local_rank={args.local_rank}, device_id={args.device_id}")
    _logger.info(f"Experiment: {experiment_name}")
    _logger.info(f"Results directory: {args.output_dir}")

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
    _logger.info(f"Optimizer: {optimizer.__class__.__name__}")

    # build loss scaler
    amp_autocast, loss_scaler = build_loss_scaler(args, _logger)
    print(f'Using amp_autocast: {amp_autocast}')
    print(f"Using loss scaler: {loss_scaler}")

    # resume from a checkpoint
    resume_epoch = None
    if args.resume and os.path.exists(args.resume):
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
        _logger.info("Using EMA model")
        if args.resume and os.path.exists(args.resume):
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # # setup distributed training
    if args.distributed:
        _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[args.device_id])
        # NOTE: EMA model does not need to be wrapped by DDP
    
    # create the train and eval dataloaders
    loader_train, loader_eval = build_dataset(args, num_aug_splits)
    
    images, targets = next(iter(loader_train))
    print(f"Training data min: {images.min()}, max: {images.max()}, mean: {images.mean()}, std: {images.std()}")
    images, targets = next(iter(loader_eval))
    print(f"Evaluation data min: {images.min()}, max: {images.max()}, mean: {images.mean()}, std: {images.std()}")


    # setup loss function
    train_loss_fn, validate_loss_fn = build_loss(args, num_aug_splits)
    print(f"Using training loss: {train_loss_fn}, validation loss: {validate_loss_fn}")
    
    # setup gradnorm regularization loss function
    reg_loss_fn = None
    gradnorm_start_epoch = args.epochs  # default: never start
    if args.gradnorm:
        reg_loss_fn = DBP(eps=args.attack_eps, std=0.225)
        gradnorm_start_epoch = args.alpha_start_epoch
        if resume_epoch is not None:
            gradnorm_start_epoch = resume_epoch
    _logger.info(f'GradNorm start: {gradnorm_start_epoch}')
    _logger.info(f'Reg losses: {str(reg_loss_fn)}')

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    print(f"Using learning rate scheduler: {lr_scheduler}")
    
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)
    
    # ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /ares
    # db_path = os.path.join(ROOT, "job_manager", "model_scheduler.db")
    # sch = Model_scheduler(db_path=db_path)

    # saver
    eval_metric = args.eval_metric
    saver = None
    best_metric = None
    best_epoch = None
    output_dir = args.output_dir
    
    if args.rank == 0:
        decreasing=True if eval_metric=='loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.max_history)
        # wandb init:
        wandb_config = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        # Use experiment_name as ID if given, otherwise generate one
        if experiment_name is None:   
            run_id = wandb.util.generate_id()
            run_name = run_id          # use run_id as the visible name
        else:
            run_id = experiment_name
            run_name = experiment_name

        wandb.init(
            project="robust-training",
            entity="ashtomer-ben-gurion-university-of-the-negev",
            id=run_id,  
            resume="allow",
            name=run_name,
            group=group_name,
            config=wandb_config,
        )
        _logger.info(f"Weights & Biases logging initialized for run: {experiment_name} in group: {group_name}")

        
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # start training
    _logger.info(f"Start training for {args.epochs-start_epoch} epochs")
    _logger.info(f"gradclip: {args.clip_grad}")
    
    already_canceled_mixup = False
    
    for epoch in range(start_epoch, args.epochs):
        if hasattr(loader_train, 'sampler') and hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)
            # mixup setting
        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            if not already_canceled_mixup:
                args.mixup_active = False
                loader_train, loader_eval = build_dataset(args, num_aug_splits)
                already_canceled_mixup = True
        # one epoch training
        train_metrics = train_one_epoch(
            epoch, model, loader_train, optimizer, train_loss_fn, args,reg_loss_fn=reg_loss_fn,
            lr_scheduler=lr_scheduler, saver=saver, amp_autocast=amp_autocast,
            loss_scaler=loss_scaler, model_ema=model_ema, _logger=_logger,gradnorm_start_epoch=gradnorm_start_epoch)

        # distributed bn sync
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        # calculate evaluation metric
        eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, _logger=_logger, epoch=epoch)

        # model ema update
        ema_eval_metrics = None
        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
            ema_eval_metrics = validate(model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)', _logger=_logger, epoch=epoch)
            
        # log wandb
        if args.rank == 0:
            wandb_log_dict = {}
            for k, v in train_metrics.items():
                wandb_log_dict[f"train/{k}"] = v
            for k, v in eval_metrics.items():
                wandb_log_dict[f"eval/{k}"] = v
            if ema_eval_metrics is not None:
                for k, v in ema_eval_metrics.items():
                    wandb_log_dict[f"eval_ema/{k}"] = v
            wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=epoch)
            wandb.log(wandb_log_dict, step=epoch)
            
        eval_metrics = ema_eval_metrics if ema_eval_metrics is not None else eval_metrics
        # lr_scheduler update
        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

        # output summary.csv
        if output_dir is not None:
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),lr=optimizer.param_groups[0]['lr'],
                write_header=best_metric is None)
            
        # save best checkpoint
        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch, eval_metrics[eval_metric])
            
        # -------- DB update: end-of-epoch --------
        # if sch is not None and args.model_id is not None and args.rank == 0:
        #     sch.update_progress_epoch_end(model_id=args.model_id, epoch=epoch+1)

        if args.distributed:
            torch.distributed.barrier()

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))



def _merge_groups_for_hydra(cfg):
    """
    Merge Hydra's grouped config dictionaries (training/model/dataset/optimizer/attacks/etc.)
    into a flat argparse.Namespace-compatible dictionary.

    This preserves ALL your original training logic WITHOUT modifying it.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    merged = {}

    # Keep all top-level primitive values
    for k, v in cfg_dict.items():
        if not isinstance(v, dict):
            merged[k] = v

    # Merge known Hydra config groups
    for group in (
        'training', 'model', 'dataset', 'optimizer',
        'attacks', 'dist', 'loss', 'lr_scheduler'
    ):
        if group in cfg_dict and isinstance(cfg_dict[group], dict):
            merged.update(cfg_dict[group])

    return merged

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def hydra_main(cfg: DictConfig):
    """
    Hydra entrypoint: converts grouped Hydra config into the argparse-style Namespace
    expected by main(args), then launches the adversarial training.
    """
    merged = _merge_groups_for_hydra(cfg)
    args = argparse.Namespace(**merged)
    main(args)


if __name__ == '__main__':
    import sys

    # Detect Hydra-style overrides such as:
    #   python adversarial_training.py training.epochs=200 optimizer.lr=1e-3
    #
    # If any CLI argument contains "=", we assume the user wants Hydra.
    if any("=" in arg for arg in sys.argv[1:]):
        hydra_main()  # use Hydra
    else:
        # ORIGINAL argparse CLI (unchanged)
        parser = argparse.ArgumentParser(
            "Robust training script",
            parents=[get_args_parser()]
        )
        args = parser.parse_args()
        opt = vars(args)

        if args.configs:
            opt.update(yaml.load(open(args.configs), Loader=yaml.FullLoader))

        args = argparse.Namespace(**opt)
        main(args)
