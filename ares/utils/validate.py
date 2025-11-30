import warnings
warnings.filterwarnings("ignore")
import time
from collections import OrderedDict
import torch

# timm functions
from timm.utils import  reduce_tensor

# robust training functions
from ares.utils.adv import adv_generator
from ares.utils.metrics import AverageMeter, accuracy

def validate(model, loader, loss_fn, args, amp_autocast=None, log_suffix='', _logger=None, epoch=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    adv_losses_m = AverageMeter()
    adv_top1_m = AverageMeter()
    adv_top5_m = AverageMeter()


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
                att_step = args.attack_step * min(epoch, 5)/5
                att_eps=args.attack_eps
                adv_input=adv_generator(args, input, target, model, att_eps, 8, att_step, random_start=True, use_best=False, attack_criterion='regular')
                with torch.no_grad():
                    with amp_autocast():
                        adv_output = model(adv_input)
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