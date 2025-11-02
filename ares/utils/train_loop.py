import warnings
warnings.filterwarnings("ignore")
import time
from collections import OrderedDict
import torch

# timm functions
from timm.models import model_parameters
from timm.utils import  reduce_tensor, dispatch_clip_grad

# robust training functions
from ares.utils.adv import adv_generator
from ares.utils.metrics import AverageMeter


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, amp_autocast=None,
        loss_scaler=None, model_ema=None, mixup_fn=None, _logger=None, writer=None, sch=None, model_id=None):
    
    # mixup setting
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        print("[DEBUG] Disabling mixup!")
    if mixup_fn is not None:
        mixup_fn.mixup_enabled = False
        print(f"[DEBUG] mixup_fn.mixup_enabled set to {mixup_fn.mixup_enabled}")

    # statistical variables
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    num_epochs = args.epochs + args.cooldown_epochs

    model.train()
    
    epoch_wall_start = time.time()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    att_step = args.attack_step * min(epoch, 5)/5
    att_eps=args.attack_eps
    if args.attack_norm=='l2':
        att_eps=att_eps*255
        att_step=att_step*255
    att_it=args.attack_it


    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx

        # processing input and target
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        if args.channels_last:
            input=input.contiguous(memory_format=torch.channels_last)
        
        data_time_m.update(time.time() - end)

        # generate adv input
        if args.advtrain:
            input_advtrain = adv_generator(args, input, target, model, att_eps, att_it, att_step, random_start=False, attack_criterion=args.attack_criterion)

        # generate advprop input
        if args.advprop:
            model.apply(lambda m: setattr(m, 'bn_mode', 'adv'))
            input_advprop = adv_generator(args, input, target, model, 1/255, 1, 1/255, random_start=True, attack_criterion=args.attack_criterion, use_best=False)
            
        # forward
        with torch.autograd.detect_anomaly():
            with amp_autocast():
                if args.advprop:
                    outputs = model(input_advprop)
                    adv_loss = loss_fn(outputs, target)
                    model.apply(lambda m: setattr(m, 'bn_mode', 'clean'))
                    outputs = model(input)
                    loss = loss_fn(outputs, target) + adv_loss
                elif args.advtrain:
                    output = model(input_advtrain)
                    loss = loss_fn(output, target)
                else:
                    output = model(input)
                    loss = loss_fn(output, target)
            # === STOP TRAINING IF LOSS IS NaN or Inf ===
            assert torch.isfinite(loss), f"Loss became non-finite at epoch {epoch}, batch {batch_idx}. Loss value: {loss.item()}"
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

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
            if writer is not None and (not args.distributed or args.rank == 0):
                global_step = num_updates  # already maintained above
                # scalars you already have handy
                writer.add_scalar('train/loss', losses_m.val, global_step)
                writer.add_scalar('train/loss_avg', losses_m.avg, global_step)
                writer.add_scalar('train/lr', lr, global_step)
                writer.add_scalar('train/time_per_batch', batch_time_m.val, global_step)
                writer.add_scalar('train/data_time', data_time_m.val, global_step)
                writer.add_scalar('train/throughput_imgs_per_s',
                                  input.size(0) * args.world_size / batch_time_m.val,
                                  global_step)
           
            _logger.info(
            'Train: [{}/{}] [{:>4d}/{} ({:>3.0f}%)]  '
            'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
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
        
    # -------- DB update: end-of-epoch --------
    if sch is not None and model_id is not None and args.rank == 0:
        sch.update_progress_epoch_end(model_id=model_id, epoch=epoch+1)
        
    return OrderedDict([('loss', losses_m.avg)])

