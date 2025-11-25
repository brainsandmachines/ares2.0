import torch.nn as nn
import torch


class DBP(nn.Module):
  def __init__(self, eps=4./255., std=0.225) -> None:
    super().__init__()
    self.eps = eps/std

  def forward(self, gradients, inputs):
    batch_size = gradients.shape[0]
    if gradients.dim() == 4:  # Typical case for image data (N, C, H, W)
      return self.eps * batch_size * gradients.abs().sum((-3, -2, -1)).mean()
    elif gradients.dim() == 2:  # Case for tabular data (N, D)
      return self.eps * batch_size * gradients.abs().sum(-1).mean()
    else:
      raise ValueError(f"Unexpected gradient dimensions: {gradients.shape}")


class GradNorm_Loss(nn.Module):
    """
    Double Backpropagation Loss (GradNorm-style).
    Implements:
        L = CE(fθ(x), y) + λ_GN * (ε / σ) * ||∇ₓ CE(fθ(x), y)||₁
    """

    def __init__(self, eps=4./255., std=0.225):
        super().__init__()
        self.eps = eps / std              # ε / σ scaling

    def forward(self, model, x, y):
        # Enable gradients w.r.t. input
        x.requires_grad_(True)

        # Forward pass and CE loss
        logits = model(x)
        loss_ce = nn.CrossEntropyLoss()(logits, y)

        # Gradient w.r.t input
        grad = torch.autograd.grad(loss_ce, x, create_graph=True)[0]

        # GradNorm penalty (L1 norm)
        if grad.dim() == 4:  # Typical case for image data (N, C, H, W)
            grad_norm = grad.abs().sum(dim=(1, 2, 3)).mean()
        elif grad.dim() == 2:  # Case for tabular data (N, D)
            grad_norm = grad.abs().sum(dim=1).mean()
        else:
            raise ValueError(f"Unexpected gradient dimensions: {grad.shape}")

        # Total loss
        loss = loss_ce + self.eps * grad_norm
        return loss



def test_loss_equivalence():
    """
    Test that running a sample batch through GradNorm loss, DBP loss, and regular
    CrossEntropy loss produces the same loss values.
    """
    # Define a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    )

    # Create a sample batch
    batch_size = 8
    inputs = torch.randn(batch_size, 10, requires_grad=True)  # Enable gradients for inputs
    targets = torch.randint(0, 3, (batch_size,))

    # Define the loss functions
    cross_entropy_loss = nn.CrossEntropyLoss()
    gradnorm_loss = GradNorm_Loss()
    dbp_loss = DBP()

    # Forward pass
    outputs = model(inputs)

    # Compute losses
    ce_loss = cross_entropy_loss(outputs, targets)
    gradnorm = gradnorm_loss(model, inputs, targets)
    gradients = torch.autograd.grad(ce_loss, inputs, create_graph=True)[0]
    dbp = dbp_loss(gradients, inputs)

    # Check equivalence
    assert gradnorm >= ce_loss, f"GradNorm loss {gradnorm} is less than CrossEntropy loss {ce_loss}"
    assert torch.isclose(gradnorm, ce_loss + gradnorm_loss.eps * dbp, atol=1e-6), (
        f"GradNorm loss {gradnorm} does not match CE + eps * DBP ({ce_loss + gradnorm_loss.eps * dbp})"
    )

    print("All loss functions produce equivalent results.")
    
    
def test_loss_combination():
    """
    Test that CrossEntropy loss + eps * DBP loss equals GradNorm loss.
    """
    # Define a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    )

    # Create a sample batch
    batch_size = 8
    inputs = torch.randn(batch_size, 10)
    targets = torch.randint(0, 3, (batch_size,))

    # Define the loss functions
    cross_entropy_loss = nn.CrossEntropyLoss()
    gradnorm_loss = GradNorm_Loss()
    dbp_loss = DBP()

    # Forward pass
    outputs = model(inputs)

    # Compute CrossEntropy loss
    ce_loss = cross_entropy_loss(outputs, targets)

    # Compute GradNorm loss
    gradnorm = gradnorm_loss(model, inputs, targets)

    # Compute DBP loss
    gradients = torch.autograd.grad(ce_loss, inputs, create_graph=True)[0]
    dbp = dbp_loss(gradients, inputs)

    # Check equivalence
    combined_loss = ce_loss + gradnorm_loss.eps * dbp
    assert torch.isclose(combined_loss, gradnorm, atol=1e-6), (
        f"Combined loss (CE + eps * DBP) {combined_loss} does not match GradNorm loss {gradnorm}"
    )

    print("CrossEntropy + eps * DBP equals GradNorm loss.")

if __name__ == "__main__":
    test_loss_equivalence()