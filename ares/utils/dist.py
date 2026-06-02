import os
import torch
import numpy as np
import random
from omegaconf import open_dict

def distributed_init(cfg):
    '''This function performs the distributed setting.'''
    dist = cfg.dist
    if dist.distributed:
        # Prefer torchrun environment variables
        with open_dict(dist):
            if 'LOCAL_RANK' in os.environ:
                dist.local_rank = int(os.environ['LOCAL_RANK'])
                dist.rank = int(os.environ.get('RANK', dist.local_rank))
                dist.world_size = int(os.environ.get('WORLD_SIZE', dist.world_size))
                dist.device_id = dist.local_rank
            elif dist.local_rank != -1:    # legacy launch flag
                dist.rank = dist.local_rank
                dist.device_id = dist.local_rank
            elif 'SLURM_PROCID' in os.environ:    # for slurm scheduler
                dist.rank = int(os.environ['SLURM_PROCID'])
                dist.device_id = dist.rank % torch.cuda.device_count()
            else:
                # single-process fallback
                dist.local_rank = 0
                dist.world_size = 1
                dist.rank = 0
                dist.device_id = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(dist.device_id)
        # With env://, let torch.distributed read RANK/WORLD_SIZE from env
        torch.distributed.init_process_group(
            backend=dist.dist_backend,
            init_method=dist.dist_url
        )
        setup_for_distributed(dist.rank == 0)
    else:
        with open_dict(dist):
            dist.local_rank = 0
            dist.world_size = 1
            dist.rank = 0
            dist.device_id = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(dist.device_id)
        
        # Initialize process group for various scenarios:
        # 1. Non-distributed scenarios (e.g., regular python script)
        # 2. torchrun with single process (where distributed=False but env vars are set)
        if not torch.distributed.is_initialized():
            if 'LOCAL_RANK' in os.environ:
                # torchrun environment - use env:// method
                torch.distributed.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method='env://'
                )
            else:
                # Single process fallback - randomize port to avoid EADDRINUSE
                
                port = int(os.environ.get("MASTER_PORT", 12000 + random.randint(0, 2000)))
                torch.distributed.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method=f'tcp://localhost:{port}',
                    world_size=1,
                    rank=0
                )
        

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print
    

def random_seed(seed=0, rank=0):
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
