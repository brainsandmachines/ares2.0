import os
import sys
import logging
import ares
from torch.utils._contextlib import _DecoratorContextManager

class CustomFormatter(logging.Formatter):
    """Class for custom formatter."""
    def format(self, record):
        """Directly output message without formattion when got 'simple' attribute."""
        if hasattr(record, 'simple') and record.simple:
            return record.getMessage()
        else:
            return logging.Formatter.format(self, record)


def setup_logger(save_dir=None, distributed_rank=0, main_only=True):
    '''Setup custom logger to record information.'''
    name = ares.__package_name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-main process
    if distributed_rank > 0 and main_only:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = CustomFormatter("[%(asctime)s %(name)s] %(levelname)s: %(message)s", '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class format_print(_DecoratorContextManager):
    '''This class is used as a decrator to format output of print func using our custom logger.'''
    def __enter__(self):
        self.pre_stdout = sys.stdout
        sys.stdout = PrintFormatter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.pre_stdout


class PrintFormatter:
    '''This class is used to overwrite the sys.stdout using our custom logger.'''
    def __init__(self):
        self.logger = logging.getLogger(name=ares.__package_name__)

    def write(self, message):
        if message != '\n':
            self.logger.info(message)

    def flush(self):
        pass
    
    
def _auto_experiment_name(args):
    group_name = "default"  # Initialize group_name with a default value
    parts = [f"{args.model}"]
    if args.advtrain:
        if args.attack_norm=="linf":
            if args.attack_criterion=="madry":
                parts.append(f"linf_{int(args.attack_eps*255)}")
                group_name = "linf_madry"
            elif args.attack_criterion=="trades":
                parts.append(f"linftrades_{int(args.attack_eps*255)}")
                group_name = "linf_trades"
            else:
                raise ValueError(f"Unknown attack criterion: {args.attack_criterion}")
        elif args.attack_norm=="l2":
            if args.attack_criterion=="madry":
                parts.append(f"l2_{args.attack_eps}")
                group_name = "l2_madry"
            elif args.attack_criterion=="trades":
                parts.append(f"l2trades_{args.attack_eps}")
                group_name = "l2_trades"
            else:
                raise ValueError(f"Unknown attack criterion: {args.attack_criterion}")
        else:
            raise ValueError(f"Unknown attack norm: {args.attack_norm}")
    if args.gradnorm:
        parts.append(f"gradnorm_{int(args.attack_eps*255)}")
        group_name = "gradnorm"
    # if args.lipshitz:
    #     parts.append(f"lip_{args.lip_coeff}")
    #     group_name = "lipshitz"
    # if args.jacobian_reg:
    #     parts.append(f"jacobian_{args.jacobian_coeff}")
    #     group_name = "jacobian"
    if len(parts)==1:
        parts.append("baseline")
        group_name = "baseline"
    if args.experiment_num is not None:
        parts.append(f"init{args.experiment_num}")
    experiment_name = "_".join(parts)

    return experiment_name, group_name
