import warnings
warnings.filterwarnings("ignore")
import time
from collections import OrderedDict
import torch

# timm functions
from timm.models import model_parameters
from timm.utils import  reduce_tensor, dispatch_clip_grad

# robust training functions
from ares.utils.adv import adv_generator, trades_adv_generator, v1_adv_generator, v1_trades_adv_generator
from ares.utils.metrics import AverageMeter
from ares.utils.gradnorm import compute_gradnorm_alpha


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, cfg,
        reg_loss_fn=None, lr_scheduler=None, saver=None, amp_autocast=None,
        loss_scaler=None, model_ema=None, _logger=None,gradnorm_start_epoch=0,
        runtime_probe=None):
    
    # statistical variables
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    grad_global_m = AverageMeter()
    grad_avg_m = AverageMeter()
    grad_max_m = AverageMeter()

    attacks = cfg.attacks
    training = cfg.training
    dataset = cfg.dataset
    optimizer_cfg = cfg.optimizer
    dist = cfg.dist
    num_epochs = training.epochs + cfg.lr_scheduler.cooldown_epochs

    model.train()
    
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    attack_domain = attacks.get('attack_domain', 'pixel')
    if attack_domain == 'pixel':
        att_step = attacks.attack_step * min(epoch, 5) / 5
        att_eps = attacks.attack_eps
        att_it = attacks.attack_it
    elif attack_domain == 'v1_feature':
        att_step = attacks.v1_attack_step * min(epoch, 5) / 5
        att_eps = attacks.v1_attack_eps
        att_it = attacks.v1_attack_it
    else:
        raise ValueError(f"Unsupported attack_domain: {attack_domain}")

    for batch_idx, (input, target) in enumerate(loader):
        if runtime_probe is not None:
            runtime_probe.maybe_start(batch_idx)
        # debugging NaN/Inf in input
        if not torch.isfinite(input).all():
            raise ValueError("Input contains NaN or Inf values")
        last_batch = batch_idx == last_idx
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        if cfg.model.channels_last:
            input=input.contiguous(memory_format=torch.channels_last)
        
        data_time_m.update(time.time() - end)

        # generate adv input
        if attacks.advtrain:
            if attacks.attack_criterion == 'madry':
                if attack_domain == 'pixel':
                    input_advtrain = adv_generator(cfg, input, target, model, att_eps, att_it, att_step, random_start=False)
                else:
                    input_advtrain = v1_adv_generator(cfg, input, target, model, att_eps, att_it, att_step, random_start=False)
            elif attacks.attack_criterion == 'trades':
                trades_random_start = attacks.attack_norm != 'l1'
                if attack_domain == 'pixel':
                    input_advtrain = trades_adv_generator(
                        cfg, input, model, att_eps, att_it, att_step, random_start=trades_random_start
                    )
                else:
                    input_advtrain = v1_trades_adv_generator(
                        cfg, input, model, att_eps, att_it, att_step, random_start=trades_random_start
                    )

        # generate advprop input
        if attacks.advprop:
            model.apply(lambda m: setattr(m, 'bn_mode', 'adv'))
            input_advprop = adv_generator(cfg, input, target, model, 1/255, 1, 1/255, random_start=True, use_best=False)
            
        # forward
        with amp_autocast():
            if attacks.advprop:
                outputs = model(input_advprop)
                adv_loss = loss_fn(outputs, target)
                model.apply(lambda m: setattr(m, 'bn_mode', 'clean'))
                outputs = model(input)
                loss = loss_fn(outputs, target) + adv_loss
            elif attacks.advtrain:
                if attacks.attack_criterion == 'madry':
                    if attack_domain == 'pixel':
                        output = model(input_advtrain)
                    else:
                        attack_model = model.module if hasattr(model, 'module') else model
                        output = attack_model.forward_from_v1_features(input_advtrain)
                    loss = loss_fn(output, target)
                elif attacks.attack_criterion == 'trades':
                    output = model(input)
                    ce_loss = loss_fn(output, target)
                    if attack_domain == 'pixel':
                        output_adv = model(input_advtrain)
                    else:
                        attack_model = model.module if hasattr(model, 'module') else model
                        output_adv = attack_model.forward_from_v1_features(input_advtrain)
                    kl_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(output_adv, dim=1),
                        torch.nn.functional.softmax(output, dim=1),
                        reduction='batchmean')
                    loss = ce_loss + attacks.trades_beta * kl_loss
            elif attacks.gradnorm:
                input.requires_grad_(True)
                output = model(input)
                ce_loss = loss_fn(output, target)
                
                gradient = torch.autograd.grad(ce_loss, input, create_graph=True, retain_graph=True)[0]
                alpha = compute_gradnorm_alpha(
                    epoch,
                    batch_idx,
                    len(loader),
                    gradnorm_start_epoch,
                    attacks.get('alpha_scale_epochs', 9.0),
                    attacks.get('alpha_scale_init', 0.1),
                )
                max_ratio = attacks.get('gradnorm_max_reg_to_ce_ratio', 0.0)
                raw_loss_reg = reg_loss_fn(gradient, input) * alpha

                if max_ratio is not None and max_ratio > 0:
                    reg_cap = max_ratio * ce_loss.detach()
                    scale = (reg_cap / raw_loss_reg.detach().clamp_min(1e-12)).clamp(max=1.0)
                    loss_reg = raw_loss_reg * scale
                else:
                    scale = raw_loss_reg.new_tensor(1.0)
                    loss_reg = raw_loss_reg

                loss = ce_loss + loss_reg
            else:
                output = model(input)
                loss = loss_fn(output, target)
        # debugging NaN/Inf in loss
        if not torch.isfinite(loss).all():
            if attacks.gradnorm:
                raise ValueError(
                    f"Loss is NaN/Inf. ce_loss={ce_loss.detach().item():.6g}, "
                    f"reg_loss={loss_reg.detach().item():.6g}, alpha={alpha:.4f}, "
                    f"epoch={epoch}, batch_idx={batch_idx}"
                )
            raise ValueError(f"Loss is NaN/Inf at epoch={epoch}, batch_idx={batch_idx}")

        if not dist.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=optimizer_cfg.clip_grad, clip_mode=optimizer_cfg.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in optimizer_cfg.clip_mode),
                create_graph=second_order)
            global_g, avg_g, max_g = compute_grad_stats(model)

        else:
            loss.backward(create_graph=second_order)
            if optimizer_cfg.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in optimizer_cfg.clip_mode),
                    value=optimizer_cfg.clip_grad, mode=optimizer_cfg.clip_mode)
            global_g, avg_g, max_g = compute_grad_stats(model)
            optimizer.step()
        
        if global_g is not None:
            grad_global_m.update(global_g)
            grad_avg_m.update(avg_g)
            grad_max_m.update(max_g)

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % cfg.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if dist.distributed:
                reduced_loss = reduce_tensor(loss.data, dist.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
           
            log_msg = (
                'Train: [{}/{}] [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Batch Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '.format(
                    epoch, num_epochs,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) * dist.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * dist.world_size / batch_time_m.avg,
                    lr=lr)
            )
            if cfg.epsilon_schedule.enabled:
                log_msg += 'Eps: {:.6g}  '.format(float(att_eps))
            log_msg += 'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(data_time=data_time_m)
            _logger.info(log_msg)
            if attacks.gradnorm and batch_idx % (cfg.log_interval * 20) == 0:
                ce_val = ce_loss.detach().item()
                raw_reg_val = raw_loss_reg.detach().item()
                reg_val = loss_reg.detach().item()
                scale_val = scale.detach().item()
                ratio = reg_val / max(ce_val, 1e-12)
                raw_ratio = raw_reg_val / max(ce_val, 1e-12)

                _logger.info(
                    'GradNorm: alpha={:.4f} ce={:.6g} raw_reg={:.6g} reg={:.6g} '
                    'raw_reg/ce={:.4f} reg/ce={:.4f} scale={:.4f}'.format(
                        alpha, ce_val, raw_reg_val, reg_val, raw_ratio, ratio, scale_val
                    )
                )

        # save checkpoint
        if saver is not None and cfg.recovery_interval and (
                last_batch or (batch_idx + 1) % cfg.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        # update lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        if runtime_probe is not None and runtime_probe.maybe_stop(batch_idx):
            break

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
        
    
        
    return OrderedDict([
    ('loss', losses_m.avg),
    ('grad_global', grad_global_m.avg),
    ('grad_avg', grad_avg_m.avg),
    ('grad_max', grad_max_m.avg),
])


def compute_grad_stats(model):
    grads = [p.grad.detach() for p in model.parameters()
             if p.requires_grad and p.grad is not None]

    if not grads:
        return None, None, None

    # global L2 norm
    global_norm = torch.sqrt(sum(g.norm()**2 for g in grads)).item()
    # average per-tensor norm
    avg_norm = sum(g.norm().item() for g in grads) / len(grads)
    # maximum element across all gradients
    max_grad = max(g.abs().max().item() for g in grads)

    return global_norm, avg_norm, max_grad
