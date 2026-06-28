import os

from omegaconf import open_dict
from timm.models import load_checkpoint

from ares.utils.epsilon_schedule import default_step_for_epsilon, denormalize_epsilon, normalize_epsilon


def active_attack_eps_field(cfg):
    return 'v1_attack_eps' if cfg.attacks.get('attack_domain', 'pixel') == 'v1_feature' else 'attack_eps'


def active_attack_step_field(cfg):
    return 'v1_attack_step' if cfg.attacks.get('attack_domain', 'pixel') == 'v1_feature' else 'attack_step'


def active_attack_it_field(cfg):
    return 'v1_attack_it' if cfg.attacks.get('attack_domain', 'pixel') == 'v1_feature' else 'attack_it'


def capture_attack_step_auto(cfg):
    with open_dict(cfg.runtime):
        cfg.runtime.active_attack_step_auto = cfg.attacks.get(active_attack_step_field(cfg), None) is None


def configure_initial_schedule_target(cfg):
    if bool(cfg.epsilon_schedule.enabled):
        target = cfg.epsilon_schedule.target_epsilon
        if target is not None:
            cfg.attacks[active_attack_eps_field(cfg)] = float(target)


def set_active_epsilon(cfg, epsilon_user):
    attack_domain = cfg.attacks.get('attack_domain', 'pixel')
    eps_internal = normalize_epsilon(float(epsilon_user), cfg.attacks.attack_norm, attack_domain=attack_domain)
    eps_field = active_attack_eps_field(cfg)
    step_field = active_attack_step_field(cfg)
    it_field = active_attack_it_field(cfg)
    cfg.attacks[eps_field] = eps_internal
    step = default_step_for_epsilon(eps_internal, cfg.attacks.attack_norm, cfg.attacks[it_field])
    if step is not None and cfg.runtime.get('active_attack_step_auto', False):
        cfg.attacks[step_field] = step
    return eps_internal


def current_active_epsilon_user(cfg):
    attack_domain = cfg.attacks.get('attack_domain', 'pixel')
    return denormalize_epsilon(
        cfg.attacks[active_attack_eps_field(cfg)],
        cfg.attacks.attack_norm,
        attack_domain=attack_domain,
    )


def load_continuation_checkpoint(model, cfg, _logger):
    path = str(cfg.continuation.checkpoint_path or '').strip()
    if not path:
        raise ValueError('continuation.checkpoint_path is required when continuation.enabled=true')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Continuation checkpoint not found: {path}')

    use_ema = bool(cfg.continuation.use_ema)
    try:
        load_checkpoint(model, path, use_ema=use_ema)
    except RuntimeError:
        if not use_ema:
            raise
        _logger.warning('Failed loading EMA weights from continuation checkpoint; falling back to state_dict.')
        load_checkpoint(model, path, use_ema=False)
    _logger.info(f'Initialized model weights from continuation checkpoint: {path}')


def maybe_save_best_adv_checkpoint(saver, epoch, eval_metrics, best_metric, best_epoch):
    if saver is None or 'advtop1' not in eval_metrics:
        return best_metric, best_epoch
    metric = eval_metrics['advtop1']
    if best_metric is None or metric > best_metric:
        saver._save(os.path.join(saver.checkpoint_dir, 'model_best_adv' + saver.extension), epoch, metric)
        return metric, epoch
    return best_metric, best_epoch
