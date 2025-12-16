import torch
import torch.nn as nn

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
    
def compute_gradnorm_alpha(epoch, batch_idx, loader_len, start_epoch):
    # progress since GradNorm started, measured in epochs
    t = (epoch - start_epoch) + batch_idx / loader_len

    if t < 0:
        return 0.0

    # increase by 0.1 per epoch
    alpha = 0.1 * (t + 1.0)

    return min(alpha, 1.0)