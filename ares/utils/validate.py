import warnings
warnings.filterwarnings("ignore")
import time
from collections import OrderedDict
import torch

# timm functions
from timm.utils import  reduce_tensor

# robust training functions
from ares.utils.adv import adv_generator, trades_adv_generator, v1_adv_generator, v1_trades_adv_generator
from ares.utils.metrics import AverageMeter, accuracy

def validate(model, loader, loss_fn, args, amp_autocast=None, log_suffix='', _logger=None, epoch=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    adv_losses_m = AverageMeter()
    adv_top1_m = AverageMeter()
    adv_top5_m = AverageMeter()
    attack_domain = getattr(args, 'attack_domain', 'pixel')

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        # read eval input
        last_batch = batch_idx == last_idx
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # normal eval process
        with torch.no_grad():
            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            # record normal results
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

        # adv eval process
        if args.advtrain or args.gradnorm:
            if epoch > 0:
                # Keep training-time warmup by default; allow eval callers to opt out.
                if attack_domain == 'pixel':
                    base_attack_step = args.attack_step
                    att_eps = args.attack_eps
                    attack_steps = int(getattr(args, "attack_it", 8))
                    attack_random_start = bool(getattr(args, "attack_random_start", False))
                    attack_use_best = bool(getattr(args, "attack_use_best", True))
                else:
                    base_attack_step = args.v1_attack_step
                    att_eps = args.v1_attack_eps
                    attack_steps = int(getattr(args, "v1_attack_it", 8))
                    attack_random_start = bool(getattr(args, "v1_attack_random_start", False))
                    attack_use_best = bool(getattr(args, "v1_attack_use_best", True))
                if getattr(args, "disable_attack_step_warmup", False):
                    att_step = base_attack_step
                else:
                    att_step = base_attack_step * min(epoch, 5) / 5
                attack_restarts = int(getattr(args, "attack_restarts", 1))
                attack_ce = torch.nn.CrossEntropyLoss(reduction='none')

                best_adv_input = None
                best_adv_loss = None
                for _ in range(max(1, attack_restarts)):
                    if args.attack_criterion == 'trades':
                        trades_random_start = args.attack_norm != 'l1'
                        if attack_domain == 'pixel':
                            cand_adv = trades_adv_generator(
                                args,
                                input,
                                model,
                                att_eps,
                                attack_steps,
                                att_step,
                                random_start=trades_random_start,
                                use_best=attack_use_best,
                            )
                        else:
                            cand_adv = v1_trades_adv_generator(
                                args,
                                input,
                                model,
                                att_eps,
                                attack_steps,
                                att_step,
                                random_start=trades_random_start,
                                use_best=attack_use_best,
                            )
                    else:
                        if attack_domain == 'pixel':
                            cand_adv = adv_generator(
                                args,
                                input,
                                target,
                                model,
                                att_eps,
                                attack_steps,
                                att_step,
                                random_start=attack_random_start,
                                use_best=attack_use_best,
                            )
                        else:
                            cand_adv = v1_adv_generator(
                                args,
                                input,
                                target,
                                model,
                                att_eps,
                                attack_steps,
                                att_step,
                                random_start=attack_random_start,
                                use_best=attack_use_best,
                            )
                    with torch.no_grad():
                        with amp_autocast():
                            if attack_domain == 'pixel':
                                cand_output = model(cand_adv)
                            else:
                                attack_model = model.module if hasattr(model, 'module') else model
                                cand_output = attack_model.forward_from_v1_features(cand_adv)
                        if isinstance(cand_output, (tuple, list)):
                            cand_output = cand_output[0]
                        cand_loss = attack_ce(cand_output, target)
                    if best_adv_loss is None:
                        best_adv_loss = cand_loss
                        best_adv_input = cand_adv
                    else:
                        replace = cand_loss > best_adv_loss
                        best_adv_input[replace] = cand_adv[replace]
                        best_adv_loss[replace] = cand_loss[replace]

                adv_input = best_adv_input
                with torch.no_grad():
                    with amp_autocast():
                        if attack_domain == 'pixel':
                            adv_output = model(adv_input)
                        else:
                            attack_model = model.module if hasattr(model, 'module') else model
                            adv_output = attack_model.forward_from_v1_features(adv_input)
                    if isinstance(adv_output, (tuple, list)):
                        adv_output = adv_output[0]
                    
                    adv_loss = loss_fn(adv_output, target)
                    adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))

                    if args.distributed:
                        adv_reduced_loss = reduce_tensor(adv_loss.data, args.world_size)
                        adv_acc1 = reduce_tensor(adv_acc1, args.world_size)
                        adv_acc5 = reduce_tensor(adv_acc5, args.world_size)
                    else:
                        adv_reduced_loss = adv_loss.data

                    torch.cuda.synchronize()

                    # record adv results
                    adv_losses_m.update(adv_reduced_loss.item(), adv_input.size(0))
                    adv_top1_m.update(adv_acc1.item(), adv_output.size(0))
                    adv_top5_m.update(adv_acc5.item(), adv_output.size(0))


        batch_time_m.update(time.time() - end)
        end = time.time()

        if last_batch or batch_idx % args.log_interval == 0:
            log_name = 'Test' + log_suffix
            _logger.info(
                '{0}: [{1:>4d}/{2}]  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                'AdvLoss: {adv_loss.val:>7.4f} ({adv_loss.avg:>6.4f})  '
                'AdvAcc@1: {adv_top1.val:>7.4f} ({adv_top1.avg:>7.4f})  '
                'AdvAcc@5: {adv_top5.val:>7.4f} ({adv_top5.avg:>7.4f})'.format(
                    log_name, batch_idx, last_idx, batch_time=batch_time_m,
                    loss=losses_m, top1=top1_m, top5=top5_m,
                    adv_loss=adv_losses_m, adv_top1=adv_top1_m, adv_top5=adv_top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg), ('advloss', adv_losses_m.avg), ('advtop1', adv_top1_m.avg), ('advtop5', adv_top5_m.avg)])
    return metrics
