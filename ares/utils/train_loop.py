import warnings
warnings.filterwarnings("ignore")
import time
from collections import OrderedDict
import torch

# timm functions
from timm.models import model_parameters
from timm.utils import  reduce_tensor, dispatch_clip_grad

# robust training functions
from ares.utils.adv import adv_generator, trades_adv_generator
from ares.utils.metrics import AverageMeter
from ares.utils.gradnorm import compute_gradnorm_alpha


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        reg_loss_fn=None, lr_scheduler=None, saver=None, amp_autocast=None,
        loss_scaler=None, model_ema=None, _logger=None,gradnorm_start_epoch=0, gradnorm_stepper=None):
    
    # statistical variables
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    grad_global_m = AverageMeter()
    grad_avg_m = AverageMeter()
    grad_max_m = AverageMeter()

    num_epochs = args.epochs + args.cooldown_epochs

    model.train()
    
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    att_step = args.attack_step * min(epoch, 5)/5
    att_eps = args.attack_eps
    att_it = args.attack_it

    for batch_idx, (input, target) in enumerate(loader):
        # debugging NaN/Inf in input
        if not torch.isfinite(input).all():
            raise ValueError("Input contains NaN or Inf values")
        last_batch = batch_idx == last_idx
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        if args.channels_last:
            input=input.contiguous(memory_format=torch.channels_last)
        
        data_time_m.update(time.time() - end)

        # generate adv input
        if args.advtrain:
            if args.attack_criterion == 'madry':
                input_advtrain = adv_generator(args, input, target, model, att_eps, att_it, att_step, random_start=False)
            elif args.attack_criterion == 'trades': 
                trades_random_start = args.attack_norm != 'l1'
                input_advtrain = trades_adv_generator(
                    args, input, model, att_eps, att_it, att_step, random_start=trades_random_start
                )

        # generate advprop input
        if args.advprop:
            model.apply(lambda m: setattr(m, 'bn_mode', 'adv'))
            input_advprop = adv_generator(args, input, target, model, 1/255, 1, 1/255, random_start=True, use_best=False)
            
        # forward
        with amp_autocast():
            if args.advprop:
                outputs = model(input_advprop)
                adv_loss = loss_fn(outputs, target)
                model.apply(lambda m: setattr(m, 'bn_mode', 'clean'))
                outputs = model(input)
                loss = loss_fn(outputs, target) + adv_loss
            elif args.advtrain:
                if args.attack_criterion == 'madry':
                    output = model(input_advtrain)
                    loss = loss_fn(output, target)
                elif args.attack_criterion == 'trades':
                    output = model(input)
                    ce_loss = loss_fn(output, target)
                    output_adv = model(input_advtrain)
                    kl_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(output_adv, dim=1),
                        torch.nn.functional.softmax(output, dim=1),
                        reduction='batchmean')
                    loss = ce_loss + args.trades_beta * kl_loss
            elif args.gradnorm:
                input.requires_grad_(True)
                output = model(input)
                ce_loss = loss_fn(output, target)
                
                gradient = torch.autograd.grad(ce_loss, input, create_graph=True, retain_graph=True)[0]
                alpha = compute_gradnorm_alpha(epoch, batch_idx, len(loader), gradnorm_start_epoch)
                loss_reg = reg_loss_fn(gradient, input)* alpha
                loss = ce_loss + loss_reg
            else:
                output = model(input)
                loss = loss_fn(output, target)
        # debugging NaN/Inf in loss
        assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
            global_g, avg_g, max_g = compute_grad_stats(model)

        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
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
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
           
            _logger.info(
            'Train: [{}/{}] [{:>4d}/{} ({:>3.0f}%)]  '
            'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
            'Batch Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
            '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
            'LR: {lr:.3e}  '
            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch, num_epochs,
                batch_idx, len(loader),
                100. * batch_idx / last_idx,
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input.size(0) * args.world_size / batch_time_m.val,
                rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                lr=lr,
                data_time=data_time_m))

        # save checkpoint
        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        # update lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

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
