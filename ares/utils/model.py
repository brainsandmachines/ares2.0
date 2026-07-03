import torch
import torch.nn as nn
from timm.models import create_model, safe_model_name
from timm.models.layers import convert_splitbn_model

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # Here we assume the color channel is in at dim=1

    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def denormalize(tensor, mean, std):
    '''
    Args:
        tensor (torch.Tensor): Float tensor image of size (B, C, H, W) to be denormalized.
        mean (torch.Tensor): float tensor means of size (C, )  for each channel.
        std (torch.Tensor): float tensor standard deviations of size (C, ) for each channel.
    '''
    return tensor*std[None]+mean[None]


def resolve_v1_gabor_seed(cfg):
    gabor_seed = cfg.model.get('v1_gabor_seed', None)
    if gabor_seed is None:
        gabor_seed = cfg.seed
    return gabor_seed

def resolve_convnext_v1_backbone(model_name):
    if not model_name.startswith('convnext_') or not model_name.endswith('_v1'):
        return None
    return model_name[:-3]

class NormalizeByChannelMeanStd(nn.Module):
    '''The class of a normalization layer.'''
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

class SwitchableBatchNorm2d(torch.nn.BatchNorm2d):
    '''The class of a batch norm layer.'''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.bn_mode = 'clean'
        self.bn_adv = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: torch.Tensor):
        if self.training:  # aux BN only relevant while training
            if self.bn_mode == 'clean':
                return super().forward(input)
            elif self.bn_mode == 'adv':
                return self.bn_adv(input)
        else:
            return super().forward(input)

def convert_switchablebn_model(module):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.

    Args:
        module (torch.nn.Module): input module
        num_splits (int): number of separate batchnorm layers to split input across
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> model = timm.models.convert_splitbn_model(model, num_splits=2)
    """

    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SwitchableBatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine,
            module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
            
        for aux in [mod.bn_adv]:
            aux.running_mean = module.running_mean.clone()
            aux.running_var = module.running_var.clone()
            aux.num_batches_tracked = module.num_batches_tracked.clone()
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_switchablebn_model(child))
    del module
    return mod

def build_model(cfg, _logger, num_aug_splits):
    '''The function to build model for robust training.'''
    # creating model
    model_cfg = cfg.model
    training = cfg.training
    dataset = cfg.dataset
    attacks = cfg.attacks
    # lazy to avoid a circular import (ares.model submodules import ares.utils.model);
    # registers custom timm models (vit_convstem, xcit_extra) before create_model
    import ares.model  # noqa: F401
    _logger.info(f"Creating model: {model_cfg.model}")
    model_kwargs = dict({
        'num_classes': model_cfg.num_classes,
        'drop_rate': training.drop,
        'drop_path_rate': training.drop_path,
        'drop_block_rate': training.drop_block,
        'global_pool': model_cfg.gp,
        'bn_momentum': model_cfg.bn_momentum,
        'bn_eps': model_cfg.bn_eps,
    })
    v1_backbone_name = resolve_convnext_v1_backbone(model_cfg.model)
    if v1_backbone_name is not None:
        from ares.model.v1_convnext import V1ConvNeXt

        model = V1ConvNeXt(
            backbone_name=v1_backbone_name,
            input_size=dataset.input_size,
            v1_noise_train_only=model_cfg.get('v1_noise_train_only', True),
            visual_degrees=model_cfg.get('v1_visual_degrees', 8),
            stride=model_cfg.get('v1_stride', 4),
            ksize=model_cfg.get('v1_ksize', 25),
            sf_corr=model_cfg.get('v1_sf_corr', 0.75),
            sf_max=model_cfg.get('v1_sf_max', 9),
            sf_min=model_cfg.get('v1_sf_min', 0),
            rand_param=model_cfg.get('v1_rand_param', False),
            gabor_seed=resolve_v1_gabor_seed(cfg),
            simple_channels=model_cfg.get('v1_simple_channels', 256),
            complex_channels=model_cfg.get('v1_complex_channels', 256),
            noise_mode=model_cfg.get('v1_noise_mode', None),
            noise_scale=model_cfg.get('v1_noise_scale', 0.35),
            noise_level=model_cfg.get('v1_noise_level', 0.07),
            k_exc=model_cfg.get('v1_k_exc', 25),
            **model_kwargs,
        )
    else:
        model = create_model(model_cfg.model, pretrained=False, **model_kwargs)
    if model_cfg.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        model_cfg.num_classes = model.num_classes
    
    _logger.info(f'Model {safe_model_name(model_cfg.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # enable split bn (separate bn stats per batch-portion)
    if model_cfg.split_bn:
        assert num_aug_splits > 1 or dataset.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # advprop conversion
    if attacks.advprop:
        model=convert_switchablebn_model(model)

    model.cuda()
    if model_cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if cfg.dist.distributed and model_cfg.sync_bn:
        if cfg.amp_version == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            from apex.parallel import convert_syncbn_model
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        _logger.info(
            'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
            'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    return model

def load_pretrained_21k(cfg, model, logger):
    '''The function to load pretrained 21K checkpoint to 1K model.'''
    logger.info(f"==============> Loading weight {cfg.model.pretrain} for fine-tuning......")
    checkpoint = torch.load(cfg.model.pretrain, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{cfg.model.pretrain}'")

    del checkpoint
    torch.cuda.empty_cache()
