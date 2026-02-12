import torch
import torch.nn as nn
import torch.distributed as dist
from torch.func import functional_call, grad_and_value

class GradNorm_Loss(nn.Module):
    """
    Double Backpropagation Loss (GradNorm-style).
    Implements:
        L = CE(fθ(x), y) + λ_GN * (ε / σ) * ||∇ₓ CE(fθ(x), y)||₁
    """

    def __init__(self, eps=4./255., std=0.225, lambda_gn=1.0):
        super().__init__()
        self.eps = eps / std              # ε / σ scaling
        self.lambda_gn = lambda_gn
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, model, x, y):
        # Enable gradients w.r.t. input
        x.requires_grad_(True)

        # Forward pass and CE loss
        logits = model(x)
        loss_ce = self.cross_entropy(logits, y)

        # Gradient w.r.t input
        grad = torch.autograd.grad(loss_ce, x, create_graph=True)[0]

        # GradNorm penalty (L1 norm)
        grad_norm = grad.abs().sum(dim=(1, 2, 3)).mean()

        # Total loss
        loss = (loss_ce
                + self.lambda_gn * self.eps * grad_norm)
        return loss
    
    
class DBP(nn.Module):
    def __init__(self, eps=4./255., std=0.225) -> None:
        super().__init__()
        self.eps = eps/std

    def forward(self, gradients, inputs):
        batch_size = gradients.shape[0]
        return self.eps*batch_size*gradients.abs().sum((-3, -2, -1)).mean()
    
def compute_gradnorm_alpha(
    epoch,
    batch_idx,
    loader_len,
    start_epoch,
    scale_epochs=9.0,
    alpha_init=0.1,
):
    # progress since GradNorm started, measured in epochs
    t = (epoch - start_epoch) + batch_idx / max(loader_len, 1)

    if t < 0:
        return 0.0

    # Scale only in the beginning, then use unscaled alpha (=1.0).
    if scale_epochs <= 0:
        return 1.0
    if t >= scale_epochs:
        return 1.0

    progress = t / scale_epochs
    alpha = alpha_init + (1.0 - alpha_init) * progress
    return min(max(alpha, 0.0), 1.0)


# --- Global Optimization Flags ---
# Enable TensorFloat32 for ~2x-3x speedup on Ampere+ GPUs (RTX 30xx/40xx, A100)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

# --- Pure Functions (Module Level) ---
# Defined globally so torch.compile can optimize them once and cache the kernel.

def _compute_loss_stateless(params, buffers, x, t, model_ref, loss_fn_ref):
    """
    Computes CE loss using explicit params/buffers.
    """
    out = functional_call(model_ref, (params, buffers), (x,))
    return loss_fn_ref(out, t)

def _total_objective_stateless(params, buffers, x, t, alpha, eps, model_ref, loss_fn_ref):
    """
    Computes (CE Loss + Regularization) and Input Gradients (dx) in one pass.
    """
    # 1. Compute CE loss AND input gradients (w.r.t arg 2 'x')
    dx, ce_loss = grad_and_value(_compute_loss_stateless, argnums=2)(
        params, buffers, x, t, model_ref, loss_fn_ref
    )
    
    # 2. GradNorm Math: Sum dims (-3, -2, -1) for (C,H,W)
    # The compiler fuses this sum with the backward pass
    grad_l1 = dx.abs().sum(dim=(-3, -2, -1)).mean()
    reg_loss = eps * x.size(0) * grad_l1
    
    return ce_loss + (reg_loss * alpha)

# Create the gradient calculator: Returns (Gradients_wrt_Params, Loss)
# argnums=0 means we differentiate w.r.t 'params'
_calc_loss_and_grads = grad_and_value(_total_objective_stateless, argnums=0)

# Compile the kernel globally
_compiled_kernel = torch.compile(_calc_loss_and_grads)


class GradNormFast:
    """
    Wrapper class to handle optimized GradNorm training.
    Encapsulates state extraction, compilation call, and gradient assignment.
    """
    def __init__(self, attack_eps=4./255., std=0.225):
        self.eps_val = attack_eps / std

    def run(self, model, loss_fn, input, target, alpha, distributed=False, world_size=1):
        """
        Executes the fused Forward+GradNorm+Backward step.
        """
        # 1. Handle DDP Models
        # If model is DDP, we want the underlying module to avoid DDP overhead 
        # inside the functional call (we handle grad sync manually).
        if hasattr(model, 'module'):
            model_ref = model.module
        else:
            model_ref = model

        # 2. Extract State (Fast pointers)
        params = dict(model_ref.named_parameters())
        buffers = dict(model_ref.named_buffers())

        # 3. Execute Compiled Kernel
        # This returns a dict of gradients {param_name: grad_tensor} and the loss value
        grads_dict, loss = _compiled_kernel(
            params, buffers, input, target, alpha, self.eps_val, model_ref, loss_fn
        )

        # 4. Assign Gradients Back to Model
        for name, p in model_ref.named_parameters():
            if name in grads_dict and grads_dict[name] is not None:
                if distributed and world_size > 1:
                    # Manually sync gradients because we bypassed DDP's backward hook
                    g = grads_dict[name]
                    dist.all_reduce(g, op=dist.ReduceOp.SUM)
                    g /= world_size
                    p.grad = g
                else:
                    p.grad = grads_dict[name]
        
        return loss