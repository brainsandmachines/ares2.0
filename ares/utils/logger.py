import os
import sys
import logging
import ares
from torch.utils._contextlib import _DecoratorContextManager

from ares.utils.epsilon_schedule import L1_EPS_MULTIPLIER, LINF_EPS_DIVISOR, denormalize_epsilon

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
    
    
def _auto_experiment_name(cfg):
    group_name = "default"  # Initialize group_name with a default value
    model_cfg = cfg.model
    attacks = cfg.attacks
    explicit_name = model_cfg.experiment_name
    if explicit_name and explicit_name != "test_experiment":
        return explicit_name, group_name

    parts = [f"{model_cfg.model}"]
    if str(model_cfg.model).startswith("convnext_") and str(model_cfg.model).endswith("_v1"):
        v1_noise_mode = model_cfg.get("v1_noise_mode", None)
        parts.append("noise" if v1_noise_mode is not None else "clean")
    dvd_cfg = cfg.dataset.get("dvd", None) if cfg.get("dataset", None) is not None else None
    if dvd_cfg is not None and bool(dvd_cfg.get("enabled", False)):
        dvd_variant = str(dvd_cfg.get("variant", "dvd-b")).replace("-", "_")
        parts.append(dvd_variant)
    if attacks.advtrain:
        attack_domain = attacks.get("attack_domain", "pixel")
        attack_eps = attacks.get("attack_eps", None)
        if attack_domain == "v1_feature":
            attack_eps = attacks.get("v1_attack_eps", attack_eps)

        if attacks.attack_norm=="linf":
            if attacks.attack_criterion=="madry":
                parts.append(f"linf_{int(attack_eps*LINF_EPS_DIVISOR)}")
                group_name = "v1feat_linf_madry" if attack_domain == "v1_feature" else "linf_madry"
            elif attacks.attack_criterion=="trades":
                parts.append(f"linftrades_{int(attack_eps*LINF_EPS_DIVISOR)}")
                group_name = "v1feat_linf_trades" if attack_domain == "v1_feature" else "linf_trades"
            else:
                raise ValueError(f"Unknown attack criterion: {attacks.attack_criterion}")
        elif attacks.attack_norm=="l2":
            l2_eps = attack_eps
            if attack_domain == "v1_feature":
                l2_eps = denormalize_epsilon(attack_eps, "l2", attack_domain=attack_domain)
            if attacks.attack_criterion=="madry":
                parts.append(f"l2_{l2_eps:g}")
                group_name = "v1feat_l2_madry" if attack_domain == "v1_feature" else "l2_madry"
            elif attacks.attack_criterion=="trades":
                parts.append(f"l2trades_{l2_eps:g}")
                group_name = "v1feat_l2_trades" if attack_domain == "v1_feature" else "l2_trades"
            else:
                raise ValueError(f"Unknown attack criterion: {attacks.attack_criterion}")
        elif attacks.attack_norm=="l1":
            if attacks.attack_criterion=="madry":
                parts.append(f"l1_{int(attack_eps/L1_EPS_MULTIPLIER)}")
                group_name = "v1feat_l1_madry" if attack_domain == "v1_feature" else "l1_madry"
            elif attacks.attack_criterion=="trades":
                parts.append(f"l1trades_{int(attack_eps/L1_EPS_MULTIPLIER)}")
                group_name = "v1feat_l1_trades" if attack_domain == "v1_feature" else "l1_trades"
            else:
                raise ValueError(f"Unknown attack criterion: {attacks.attack_criterion}")
        else:
            raise ValueError(f"Unknown attack norm: {attacks.attack_norm}")
    if attacks.gradnorm:
        gradnorm_penalty_norm = attacks.get("gradnorm_penalty_norm", "l1")
        parts.append(f"gradnorm_{gradnorm_penalty_norm}_{int(attacks.attack_eps*255)}")
        group_name = f"gradnorm_{gradnorm_penalty_norm}"
    # if cfg.lipshitz:
    #     parts.append(f"lip_{cfg.lip_coeff}")
    #     group_name = "lipshitz"
    # if cfg.jacobian_reg:
    #     parts.append(f"jacobian_{cfg.jacobian_coeff}")
    #     group_name = "jacobian"
    if len(parts)==1:
        parts.append("baseline")
        group_name = "baseline"
    if model_cfg.experiment_num is not None:
        parts.append(f"init{model_cfg.experiment_num}")
    experiment_name = "_".join(parts)

    return experiment_name, group_name
