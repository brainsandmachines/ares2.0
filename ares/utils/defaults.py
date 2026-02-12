import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Robust training script', add_help=False)
    parser.add_argument('--configs', default='', type=str)

    #* distributed setting
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--local-rank','--local_rank',default=-1, type=int)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--rank', default=-1, type=int, help='rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-backend', default='nccl', help='backend used to set up distributed training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    #* amp parameters
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--amp_version', default='', help='amp version')

    #* model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--num-classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain', default='', help='pretrain from checkpoint')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None. (opt)')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout (opt)')
    
    parser.add_argument('--model_id', type=str,default='unknown_model',   help='Identifier for the model configuration.')
    parser.add_argument('--experiment_name', type=str, default='test_experiment', help='Name for the experiment.')
    #* Batch norm parameters
    parser.add_argument('--bn-momentum', type=float, default=None, help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true', default=False, help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce', help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true', default=False,
                        help='Enable separate BN layers per augmentation split.')

    #* Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer-decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')

    #* Learning rate schedule parameters
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--sched-on-updates', action='store_true', default=False,
                    help='Apply LR scheduler step on update instead of epoch end.')
    parser.add_argument('--lrb', type=float, default=0.1, metavar='LR',
                        help='base learning rate (default: 5e-4)')
    parser.add_argument('--lr', type=float, default=None, help='actual learning rate')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    #* dataset parameters
    parser.add_argument('--batch-size', default=64, type=int)    # batch size per gpu
    parser.add_argument('--train-dir', default='', type=str, help='train dataset path')
    parser.add_argument('--eval-dir', default='', type=str, help='validation dataset path')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--crop-pct', default=0.875, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=(0.229, 0.224, 0.225), metavar='STD',
                        help='Override std deviation of of dataset')
    
    #* Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-repeats', type=float, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    # random erase
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    # drop connection
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    #* ema
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--max-history', type=int, default=5, help='how many recovery checkpoints')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1")')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # advtrain
    parser.add_argument('--advtrain', default=False, help='if use advtrain')
    parser.add_argument('--attack-criterion', type=str, default='regular', choices=['regular', 'smooth', 'mixup'], help='default args for: adversarial training')
    parser.add_argument('--attack-eps', type=float, default=2.0, help='attack epsilon.')
    parser.add_argument('--attack-step', type=float, default=8.0/3, help='attack epsilon.')
    parser.add_argument('--attack-it', type=int, default=3, help='attack iteration')
    parser.add_argument('--attack-norm',type=str, default='linf',choices=['linf','l2'], help='norm used for adversarial training step')
    parser.add_argument('--alpha-start-epoch', type=int, default=0, help='epoch to start gradnorm objective')
    parser.add_argument('--alpha-scale-init', type=float, default=0.1, help='initial gradnorm alpha during scaling phase')
    parser.add_argument('--alpha-scale-epochs', type=float, default=9.0, help='number of early epochs to scale gradnorm alpha before fixing alpha=1.0')
    parser.add_argument('--gradnorm-max-reg-to-ce-ratio', type=float, default=3.0, help='caps gradnorm regularization relative to CE loss; <=0 disables cap')
    # advprop
    parser.add_argument('--advprop', default=False, help='if use advprop')
    parser.add_argument('--experiment', default='', type=str,
                        help='Experiment name for output directory. '
                             'If empty, auto = "<model>_eps<eps255>_<norm>".')

    # final evaluation (AutoAttack + PGD sweep)
    parser.add_argument('--final-eval', action='store_true', default=False,
                        help='Run final evaluation after training (AutoAttack and/or PGD sweep)')
    parser.add_argument('--final-eval-ckpt-name', default='model_best.pth.tar',
                        help='Checkpoint filename under output_dir to evaluate')
    parser.add_argument('--final-eval-autoattack', action='store_true', default=False,
                        help='Run AutoAttack in final evaluation')
    parser.add_argument('--final-eval-aa-batch-size', type=int, default=None,
                        help='Batch size for AutoAttack evaluation (default: training batch_size)')
    parser.add_argument('--final-eval-aa-norm', default=None, choices=['Linf', 'L2'],
                        help='Override AutoAttack norm (requires --final-eval-aa-eps)')
    parser.add_argument('--final-eval-aa-eps', type=float, default=None,
                        help='Override AutoAttack epsilon (requires --final-eval-aa-norm)')
    parser.add_argument('--final-eval-aa-max-batches', type=int, default=None,
                        help='Limit AutoAttack to N batches (debug)')
    parser.add_argument('--final-eval-pgd', action='store_true', default=False,
                        help='Run PGD epsilon sweep in final evaluation')
    parser.add_argument('--final-eval-pgd-eps', default='0.5,1,2,4,8,16',
                        help='Comma-separated eps list for PGD sweep')
    parser.add_argument('--final-eval-pgd-norms', default='linf,l2,l1',
                        help='Comma-separated norms for PGD sweep (linf,l2,l1)')
    parser.add_argument('--final-eval-pgd-attack-steps', type=int, default=10,
                        help='PGD attack steps for sweep')
    parser.add_argument('--final-eval-pgd-batch-size', type=int, default=None,
                        help='Batch size for PGD sweep evaluation (default: training batch_size)')
    parser.add_argument('--final-eval-pgd-max-batches', type=int, default=None,
                        help='Limit PGD sweep to N batches (debug)')
    parser.add_argument('--final-eval-plots', action='store_true', default=False,
                        help='Generate accuracy-vs-epsilon plots for PGD sweep')
    parser.add_argument('--final-eval-plot-x-col', default='epsilon_input',
                        choices=['epsilon_input', 'epsilon_eval'],
                        help='X-axis column for PGD plots')
    parser.add_argument('--final-eval-out-dir', default='',
                        help='Output directory for final evaluation (default: <output_dir>)')
    parser.add_argument('--final-eval-val-dir', default='',
                        help='Validation dir for final evaluation (default: eval-dir)')
    parser.add_argument('--final-eval-num-workers', type=int, default=8,
                        help='Num workers for final evaluation DataLoaders')

    return parser
