import logging
import warnings
warnings.filterwarnings("ignore")
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
import os
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import wandb

# timm functions
from timm.models import resume_checkpoint, load_checkpoint
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import ModelEmaV2, distribute_bn, CheckpointSaver, update_summary

# robust training functions
from ares.utils.dist import distributed_init, random_seed
from ares.utils.logger import setup_logger , _auto_experiment_name
from ares.utils.model import build_model
from ares.utils.loss import build_loss, resolve_amp, build_loss_scaler
from ares.utils.gradnorm import DBP 
from ares.utils.dataset import build_dataset
from ares.utils.train_loop import train_one_epoch
from ares.utils.validate import validate
from ares.utils.runtime_probe import PhaseTimer, append_runtime_row
from ares.utils.final_eval_helpers import _maybe_run_final_eval
from ares.utils.epsilon_schedule import L1_EPS_MULTIPLIER, LINF_EPS_DIVISOR, build_epsilon_schedule, normalize_epsilon
from ares.utils.continuation import (
    active_attack_eps_field,
    capture_attack_step_auto,
    configure_initial_schedule_target,
    current_active_epsilon_user,
    load_continuation_checkpoint,
    maybe_save_best_adv_checkpoint,
    set_active_epsilon,
)

from orchestrator import progress as orch_progress
from aircc.aircc_job_manager import progress as aircc_progress

def main(cfg: DictConfig):
    # distributed settings and logger
    if "WORLD_SIZE" in os.environ:
        with open_dict(cfg.dist):
            cfg.dist.world_size = int(os.environ["WORLD_SIZE"])
    with open_dict(cfg.dist):
        cfg.dist.distributed = float(cfg.dist.world_size) > 1
    with open_dict(cfg):
        cfg.runtime = {}
        cfg.hydra_config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    distributed_init(cfg)

    # aircc job manager: cap this process to a fraction of GPU memory so 2 parallel
    # training procs can share one B200 without OOM-ing each other. No-op if unset.
    _aircc_mem_frac = os.environ.get("AIRCC_MEM_FRACTION")
    if _aircc_mem_frac and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(float(_aircc_mem_frac), cfg.dist.device_id)
        except Exception:
            pass

    attack_domain = cfg.attacks.get('attack_domain', 'pixel')
    v1_noise_mode = cfg.model.get('v1_noise_mode', None)
    is_convnext_v1 = str(cfg.model.model).startswith('convnext_') and str(cfg.model.model).endswith('_v1')

    if (
        is_convnext_v1
        and cfg.attacks.advtrain
        and cfg.attacks.attack_criterion in {'madry', 'trades'}
        and v1_noise_mode is not None
    ):
        raise ValueError('Adversarial training with V1 noise is not supported. Set v1_noise_mode=null.')

    if bool(cfg.epsilon_schedule.enabled):
        capture_attack_step_auto(cfg)
        configure_initial_schedule_target(cfg)

    # Normalize pixel-space attack magnitudes and derive norm-specific defaults.
    if attack_domain == 'pixel':
        if cfg.attacks.attack_norm == 'linf':
            if cfg.attacks.attack_eps is not None:
                cfg.attacks.attack_eps = float(cfg.attacks.attack_eps) / LINF_EPS_DIVISOR
            if cfg.attacks.attack_step is not None:
                cfg.attacks.attack_step = float(cfg.attacks.attack_step) / LINF_EPS_DIVISOR
        elif cfg.attacks.attack_norm == 'l1':
            if cfg.attacks.attack_step is None:
                cfg.attacks.attack_step = 1.0
            if cfg.attacks.attack_eps is not None:
                cfg.attacks.attack_eps = float(cfg.attacks.attack_eps) * L1_EPS_MULTIPLIER

        if (
            cfg.attacks.attack_norm == 'linf'
            and cfg.attacks.attack_step is None
            and cfg.attacks.attack_eps is not None
        ):
            cfg.attacks.attack_step = float(cfg.attacks.attack_eps) / max(int(cfg.attacks.attack_it), 1)
        elif (
            cfg.attacks.attack_norm == 'l2'
            and cfg.attacks.attack_step is None
            and cfg.attacks.attack_eps is not None
        ):
            cfg.attacks.attack_step = 2.0 * float(cfg.attacks.attack_eps) / max(int(cfg.attacks.attack_it), 1)
    else:
        if cfg.attacks.attack_norm == 'linf':
            if cfg.attacks.v1_attack_eps is not None:
                cfg.attacks.v1_attack_eps = float(cfg.attacks.v1_attack_eps) / LINF_EPS_DIVISOR
            if cfg.attacks.v1_attack_step is not None:
                cfg.attacks.v1_attack_step = float(cfg.attacks.v1_attack_step) / LINF_EPS_DIVISOR
        elif cfg.attacks.attack_norm == 'l1':
            if cfg.attacks.v1_attack_step is None:
                cfg.attacks.v1_attack_step = 1.0
            if cfg.attacks.v1_attack_eps is not None:
                cfg.attacks.v1_attack_eps = float(cfg.attacks.v1_attack_eps) * L1_EPS_MULTIPLIER
        elif cfg.attacks.attack_norm == 'l2':
            if cfg.attacks.v1_attack_eps is not None:
                cfg.attacks.v1_attack_eps = normalize_epsilon(
                    float(cfg.attacks.v1_attack_eps),
                    cfg.attacks.attack_norm,
                    attack_domain=attack_domain,
                )
        if (
            cfg.attacks.attack_norm == 'linf'
            and cfg.attacks.v1_attack_step is None
            and cfg.attacks.v1_attack_eps is not None
        ):
            cfg.attacks.v1_attack_step = float(cfg.attacks.v1_attack_eps) / max(int(cfg.attacks.v1_attack_it), 1)
        elif (
            cfg.attacks.attack_norm == 'l2'
            and cfg.attacks.v1_attack_step is None
            and cfg.attacks.v1_attack_eps is not None
        ):
            cfg.attacks.v1_attack_step = 2.0 * float(cfg.attacks.v1_attack_eps) / max(int(cfg.attacks.v1_attack_it), 1)

    if cfg.dist.rank == 0:
        experiment_name, group_name = _auto_experiment_name(cfg)
        cfg.output_dir = os.path.join(cfg.output_dir, experiment_name)
        os.makedirs(cfg.output_dir, exist_ok = True)
    
    _logger = setup_logger(save_dir=cfg.output_dir, distributed_rank=cfg.dist.rank)
    _logger.info(f"Runtime distributed={cfg.dist.distributed}, world_size={cfg.dist.world_size}, rank={cfg.dist.rank}, local_rank={cfg.dist.local_rank}, device_id={cfg.dist.device_id}")
    _logger.info(f"Experiment: {experiment_name}")
    _logger.info(f"Results directory: {cfg.output_dir}")

    # fix the seed for reproducibility
    if cfg.seed is None:
        cfg.seed = int(torch.randint(0, 2**32 - 1, ()).item())
    _logger.info(f"Using seed: {cfg.seed}")
    random_seed(cfg.seed, cfg.dist.rank)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark = True
    
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if cfg.dataset.aug_splits > 0:
        assert cfg.dataset.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = cfg.dataset.aug_splits
    
    # resolve amp
    resolve_amp(cfg, _logger)

    # build model
    model = build_model(cfg, _logger, num_aug_splits)

    # create optimizer
    optimizer=None
    if cfg.optimizer.lr is None:
        cfg.optimizer.lr = cfg.lr_scheduler.lrb * cfg.training.batch_size * cfg.dist.world_size / 512
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=cfg.optimizer))
    _logger.info(f"Optimizer: {optimizer.__class__.__name__}")

    # build loss scaler
    amp_autocast, loss_scaler = build_loss_scaler(cfg, _logger)
    print(f'Using amp_autocast: {amp_autocast}')
    print(f"Using loss scaler: {loss_scaler}")

    # resume from a checkpoint
    resume_epoch = None
    if cfg.model.resume and os.path.exists(cfg.model.resume):
        resume_epoch = resume_checkpoint(
            model, cfg.model.resume,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            log_info=cfg.dist.rank == 0)
    elif bool(cfg.continuation.enabled):
        load_continuation_checkpoint(model, cfg, _logger)

    # setup ema
    model_ema = None
    if cfg.training.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=cfg.training.model_ema_decay, device='cpu' if cfg.training.model_ema_force_cpu else None)
        _logger.info("Using EMA model")
        if cfg.model.resume and os.path.exists(cfg.model.resume):
            load_checkpoint(model_ema.module, cfg.model.resume, use_ema=True)

    if cfg.model.get('compile_model', False):
      if not hasattr(torch, 'compile'):
          raise RuntimeError('compile_model=True requires torch.compile, available in PyTorch 2.x')
      _logger.info('Compiling model with torch.compile')
      model = torch.compile(model)

    # # setup distributed training
    if cfg.dist.distributed:
        _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[cfg.dist.device_id])
        # NOTE: EMA model does not need to be wrapped by DDP
    
    # create the train and eval dataloaders
    loader_train, loader_eval = build_dataset(cfg, num_aug_splits)
    
    images, targets = next(iter(loader_train))
    print(f"Training data min: {images.min()}, max: {images.max()}, mean: {images.mean()}, std: {images.std()}")
    images, targets = next(iter(loader_eval))
    print(f"Evaluation data min: {images.min()}, max: {images.max()}, mean: {images.mean()}, std: {images.std()}")


    # setup loss function
    train_loss_fn, validate_loss_fn = build_loss(cfg, num_aug_splits)
    print(f"Using training loss: {train_loss_fn}, validation loss: {validate_loss_fn}")
    
    # setup gradnorm regularization loss function
    reg_loss_fn = None
    gradnorm_start_epoch = cfg.training.epochs  # default: never start
    if cfg.attacks.gradnorm:
        reg_loss_fn = DBP(
            eps=cfg.attacks.attack_eps,
            std=0.225,
            penalty_norm=cfg.attacks.get('gradnorm_penalty_norm', 'l1'),
        )
        gradnorm_start_epoch = cfg.attacks.alpha_start_epoch
    _logger.info(f'GradNorm start: {gradnorm_start_epoch}')
    _logger.info(
        f"GradNorm penalty norm: {cfg.attacks.get('gradnorm_penalty_norm', 'l1')}, "
        f"GradNorm alpha scaling: init={cfg.attacks.get('alpha_scale_init', 0.1)}, "
        f"scale_epochs={cfg.attacks.get('alpha_scale_epochs', 9.0)}, "
        f"max_reg_to_ce_ratio={cfg.attacks.get('gradnorm_max_reg_to_ce_ratio', 3.0)}"
    )
    _logger.info(f'Reg losses: {str(reg_loss_fn)}')
    
    if cfg.attacks.attack_criterion == 'trades' and cfg.attacks.advtrain:
        cfg.attacks.random_start = True

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    scheduler_cfg = OmegaConf.to_container(cfg.lr_scheduler, resolve=True)
    scheduler_cfg.update({
        "epochs": cfg.training.epochs,
        "seed": cfg.seed,
        "eval_metric": cfg.eval_metric,
    })
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(OmegaConf.create(scheduler_cfg)),
        updates_per_epoch=updates_per_epoch,
    )
    print(f"Using learning rate scheduler: {lr_scheduler}")
    
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if cfg.lr_scheduler.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)
    
    # ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /ares
    # db_path = os.path.join(ROOT, "job_manager", "model_scheduler.db")
    # sch = Model_scheduler(db_path=db_path)

    # saver
    eval_metric = cfg.eval_metric
    saver = None
    best_metric = None
    best_epoch = None
    best_adv_metric = None
    best_adv_epoch = None
    output_dir = cfg.output_dir
    epsilon_schedule = build_epsilon_schedule(cfg)
    target_epsilon_user = None
    target_epsilon_internal = None
    if epsilon_schedule is not None:
        target_epsilon_user = float(epsilon_schedule.target_epsilon)
        target_epsilon_internal = normalize_epsilon(
            target_epsilon_user,
            cfg.attacks.attack_norm,
            attack_domain=attack_domain,
        )
    
    if cfg.dist.rank == 0:
        decreasing=True if eval_metric=='loss' else False
        checkpoint_metadata = OmegaConf.create(
            {
                "model": cfg.model.model,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
        )
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=checkpoint_metadata, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=cfg.model.max_history)
        # wandb init:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        # Use experiment_name as ID if given, otherwise generate one
        if experiment_name is None:   
            run_id = wandb.util.generate_id()
            run_name = run_id          # use run_id as the visible name
        else:
            run_id = experiment_name
            run_name = experiment_name

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "robust-training"),
            entity="ashtomer-ben-gurion-university-of-the-negev",
            id=run_id,  
            resume="allow",
            name=run_name,
            group=group_name,
            config=wandb_config,
        )
        _logger.info(f"Weights & Biases logging initialized for run: {experiment_name} in group: {group_name}")

        
        with open(os.path.join(output_dir, 'runtime_config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        with open(os.path.join(output_dir, 'hydra_config.yaml'), 'w') as f:
            f.write(cfg.hydra_config_yaml)

    did_train_this_run = start_epoch < cfg.training.epochs

    # start training
    _logger.info(f"Start training for {cfg.training.epochs-start_epoch} epochs")
    _logger.info(f"gradclip: {cfg.optimizer.clip_grad}")
    
    already_canceled_mixup = False
    
    for epoch in range(start_epoch, cfg.training.epochs):
        if epsilon_schedule is not None:
            train_epsilon_user = epsilon_schedule.value_for_epoch(epoch)
            train_epsilon_internal = set_active_epsilon(cfg, train_epsilon_user)
            _logger.info(
                f"Epsilon schedule epoch={epoch + 1}: train_epsilon={train_epsilon_user:g} "
                f"(internal={train_epsilon_internal:.8g})"
            )
        else:
            train_epsilon_user = current_active_epsilon_user(cfg)
            train_epsilon_internal = cfg.attacks[active_attack_eps_field(cfg)]
            target_epsilon_user = train_epsilon_user
            target_epsilon_internal = train_epsilon_internal

        if hasattr(loader_train, 'sampler') and hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)
            # mixup setting
        if cfg.dataset.mixup_off_epoch and epoch >= cfg.dataset.mixup_off_epoch:
            if not already_canceled_mixup:
                cfg.dataset.mixup_active = False
                loader_train, loader_eval = build_dataset(cfg, num_aug_splits)
                already_canceled_mixup = True
        # one epoch training
        if cfg.runtime_probe:
            train_pt = PhaseTimer(cfg.runtime_probe_warmup_batches, cfg.runtime_probe_measure_batches, len(loader_train))
        else:
            train_pt = None
        train_metrics = train_one_epoch(
            epoch, model, loader_train, optimizer, train_loss_fn, cfg,reg_loss_fn=reg_loss_fn,
            lr_scheduler=lr_scheduler, saver=saver, amp_autocast=amp_autocast,
            loss_scaler=loss_scaler, model_ema=model_ema, _logger=_logger,gradnorm_start_epoch=gradnorm_start_epoch,
            runtime_probe=train_pt)
        if epsilon_schedule is not None:
            train_metrics['epsilon'] = float(train_epsilon_user)
            train_metrics['epsilon_internal'] = float(train_epsilon_internal)

        if epsilon_schedule is not None:
            target_epsilon_internal = set_active_epsilon(cfg, target_epsilon_user)

        # distributed bn sync
        if cfg.dist.distributed and cfg.model.dist_bn in ('broadcast', 'reduce'):
            _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, cfg.dist.world_size, cfg.model.dist_bn == 'reduce')

        # calculate evaluation metric
        if cfg.runtime_probe:
            eval_pt = PhaseTimer(cfg.runtime_probe_warmup_batches, cfg.runtime_probe_measure_batches, len(loader_eval))
        else:
            eval_pt = None
        eval_metrics = validate(model, loader_eval, validate_loss_fn, cfg, amp_autocast=amp_autocast, _logger=_logger, epoch=epoch, runtime_probe=eval_pt)
        if epsilon_schedule is not None:
            eval_metrics['epsilon'] = float(target_epsilon_user)
            eval_metrics['epsilon_internal'] = float(target_epsilon_internal)

        # model ema update
        ema_eval_metrics = None
        if model_ema is not None and not cfg.training.model_ema_force_cpu:
            if cfg.dist.distributed and cfg.model.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, cfg.dist.world_size, cfg.model.dist_bn == 'reduce')
            ema_eval_metrics = validate(model_ema.module, loader_eval, validate_loss_fn, cfg, amp_autocast=amp_autocast, log_suffix=' (EMA)', _logger=_logger, epoch=epoch)
            if epsilon_schedule is not None:
                ema_eval_metrics['epsilon'] = float(target_epsilon_user)
                ema_eval_metrics['epsilon_internal'] = float(target_epsilon_internal)
            
        # log wandb
        if cfg.dist.rank == 0:
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
            
        # runtime probe: record timing row and skip checkpointing (timing-only run)
        if cfg.runtime_probe:
            if cfg.dist.rank == 0:
                train_runtime = train_pt.full_estimate()
                eval_runtime = eval_pt.full_estimate()
                append_runtime_row(cfg.runtime_probe_csv, {
                    'protocol': cfg.runtime_probe_protocol,
                    'arch': cfg.runtime_probe_arch,
                    'bsz': cfg.training.batch_size,
                    'optimizer': cfg.optimizer.opt,
                    'compile': bool(cfg.model.compile_model),
                    'fullepochruntime': round(train_runtime + eval_runtime, 3),
                    'train_runtime': round(train_runtime, 3),
                    'eval_runtime': round(eval_runtime, 3),
                })
                _logger.info(
                    f"[runtime_probe] {cfg.runtime_probe_protocol}/{cfg.runtime_probe_arch} "
                    f"train={train_runtime:.1f}s eval={eval_runtime:.1f}s "
                    f"full={train_runtime + eval_runtime:.1f}s -> {cfg.runtime_probe_csv}")
            continue

        # save best checkpoint
        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch, eval_metrics[eval_metric])
            if bool(cfg.checkpointing.save_best_adv) and (cfg.attacks.advtrain or cfg.attacks.gradnorm):
                best_adv_metric, best_adv_epoch = maybe_save_best_adv_checkpoint(
                    saver, epoch, eval_metrics, best_adv_metric, best_adv_epoch
                )
            
        # -------- orchestrator: atomic end-of-epoch progress (no-op if not tracked) --------
        orch_progress.update_epoch(epoch + 1, rank=cfg.dist.rank)
        # -------- aircc job manager: end-of-epoch progress (no-op if not tracked) --------
        aircc_progress.update_epoch(
            epoch + 1, rank=cfg.dist.rank,
            checkpoint=os.path.join(output_dir, 'last.pth.tar') if output_dir is not None else None,
        )

        if cfg.dist.distributed:
            torch.distributed.barrier()

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    if best_adv_metric is not None:
        _logger.info('*** Best adv metric: {0} (epoch {1})'.format(best_adv_metric, best_adv_epoch))

    if cfg.dist.rank == 0:
        _maybe_run_final_eval(cfg, output_dir, _logger, did_train_this_run)
        # orchestrator: a pure-training invocation (final_eval off) has finished
        # its run -> hand the row off to the AutoAttack stage. No-op when the job
        # is untracked (manual run) or when this is itself the AA invocation
        # (final_eval=True), whose PLOTTING handoff lives in final_eval_helpers.
        if not bool(cfg.get("final_eval", False)):
            orch_progress.advance_status("AA_EVAL", rank=cfg.dist.rank)

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def hydra_main(cfg: DictConfig):
    """Hydra entrypoint for adversarial training."""
    main(cfg)


if __name__ == '__main__':
    hydra_main()
