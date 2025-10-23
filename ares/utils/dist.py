import os
import torch
import numpy as np
import random

def distributed_init(args):
    '''This function performs the distributed setting.'''
    if args.distributed:
        # Prefer torchrun environment variables
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
            args.rank = int(os.environ.get('RANK', args.local_rank))
            args.world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
            args.device_id = args.local_rank
        elif args.local_rank != -1:    # legacy launch flag
            args.rank = args.local_rank
            args.device_id = args.local_rank
        elif 'SLURM_PROCID' in os.environ:    # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.device_id = args.rank % torch.cuda.device_count()
        else:
            # single-process fallback
            args.local_rank = 0
            args.world_size = 1
            args.rank = 0
            args.device_id = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(args.device_id)
        # With env://, let torch.distributed read RANK/WORLD_SIZE from env
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url
        )
        setup_for_distributed(args.rank == 0)
    else:
        args.local_rank = 0
        args.world_size = 1
        args.rank = 0
        args.device_id = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(args.device_id)
        
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
