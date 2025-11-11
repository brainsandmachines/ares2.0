# tests/test_attacker_steps.py
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from ares.utils.adv import L2Step, LinfStep, adv_generator

torch.manual_seed(0)


def per_sample_l2_norm(x, y):
    """Compute per-sample L2 norm of (x - y). Returns tensor shape [B]."""
    diff = (x - y).view(x.shape[0], -1)
    return torch.norm(diff, dim=1)


def per_sample_linf_norm(x, y):
    """Compute per-sample L-inf norm of (x - y). Returns tensor shape [B]."""
    diff = (x - y).view(x.shape[0], -1)
    return torch.max(torch.abs(diff), dim=1).values


@pytest.mark.parametrize("B,C,H,W", [(1, 3, 32, 32), (4, 3, 16, 16)])
@pytest.mark.parametrize("eps_l2", [1.0, 3.0])
def test_l2_step_shapes_and_projection(B, C, H, W, eps_l2):
    """
    L2Step should:
    - accept [B,C,H,W] inputs and return same-shaped tensors
    - project outputs so per-sample L2 distance from orig_input <= eps_l2 (+ tiny tol)
    """
    device = torch.device("cpu")

    # random "original" image in [0,1]
    orig = torch.rand(B, C, H, W, device=device)
    step_size = 0.5  # arbitrary
    l2step = L2Step(orig_input=orig, eps=eps_l2, step_size=step_size, use_grad=True)

    # 1) random_perturb -> shape & projection
    rand_x = l2step.random_perturb(orig.clone())
    assert rand_x.shape == (B, C, H, W)
    norms = per_sample_l2_norm(rand_x, orig)
    assert torch.all(norms <= eps_l2 + 1e-5), f"L2 random_perturb produced norms {norms} > {eps_l2}"

    # 2) step with a fake gradient -> projection after project()
    fake_grad = torch.randn_like(orig)
    stepped = l2step.step(rand_x.clone(), fake_grad)
    assert stepped.shape == (B, C, H, W)

    projected = l2step.project(stepped)
    assert projected.shape == (B, C, H, W)

    norms = per_sample_l2_norm(projected, orig)
    assert torch.all(norms <= eps_l2 + 1e-5), f"L2 project allowed norms {norms} > {eps_l2}"

    # 3) ensure projection doesn't change tensors outside allowed diag (sanity)
    # If input already within eps, projecting should keep it within eps (monotonic)
    inside = orig + (torch.randn_like(orig).view(B, -1) * 0.0).view_as(orig)  # exactly orig
    proj_inside = l2step.project(inside)
    norms2 = per_sample_l2_norm(proj_inside, orig)
    assert torch.all(norms2 <= eps_l2 + 1e-5)


@pytest.mark.parametrize("B,C,H,W", [(1, 3, 32, 32), (4, 3, 16, 16)])
@pytest.mark.parametrize("eps_px", [2.0, 4.0])  # eps in pixels (we'll divide by 255)
def test_linf_step_shapes_and_projection(B, C, H, W, eps_px):
    """
    LinfStep should:
    - accept [B,C,H,W] inputs and return same-shaped tensors
    - projection keeps per-sample Linf distance <= eps_px/255 (+ tiny tol)
    """
    device = torch.device("cpu")

    # construct original image in [0,1]
    orig = torch.rand(B, C, H, W, device=device)
    eps = eps_px / 255.0
    step_size = 1.0 / 255.0
    linf = LinfStep(orig_input=orig, eps=eps, step_size=step_size, use_grad=True)

    # random_perturb -> shape & projection
    rand_x = linf.random_perturb(orig.clone())
    assert rand_x.shape == (B, C, H, W)
    norms = per_sample_linf_norm(rand_x, orig)
    assert torch.all(norms <= eps + 1e-6), f"Linf random_perturb produced norms {norms} > {eps}"

    # step with fake gradient -> project -> check
    fake_grad = torch.randn_like(orig)
    stepped = linf.step(rand_x.clone(), fake_grad)
    assert stepped.shape == (B, C, H, W)

    projected = linf.project(stepped)
    assert projected.shape == (B, C, H, W)

    norms = per_sample_linf_norm(projected, orig)
    assert torch.all(norms <= eps + 1e-6), f"Linf project allowed norms {norms} > {eps}"

    # random_uniform if provided should also be within the same bound
    if hasattr(linf, "random_uniform"):
        ru = linf.random_uniform(orig.clone())
        assert ru.shape == (B, C, H, W)
        norms_ru = per_sample_linf_norm(ru, orig)
        assert torch.all(norms_ru <= eps + 1e-6), f"Linf random_uniform produced norms {norms_ru} > {eps}"


# -----------------------------
# Integration tests for adv_generator (L2 and Linf)
# These tests run on CUDA if available. If CUDA is not available, we monkeypatch
# torch.Tensor.cuda to a no-op temporarily so that adv_generator's internal
# `.cuda()` calls do not raise errors and the test still runs on CPU.
# -----------------------------

def _run_adv_generator_test(args, images_norm, target, model, eps, attack_steps, attack_lr, random_start):
    """
    Helper: runs adv_generator while handling .cuda() calls internally.
    If CUDA available, move tensors & model to cuda and run normally.
    If no CUDA, monkeypatch torch.Tensor.cuda to identity for the call.
    Returns the resulting adv tensor (on the same device as images_norm input).
    """
    if torch.cuda.is_available():
        # run on GPU
        device = torch.device("cuda")
        model = model.cuda()
        images = images_norm.cuda()
        target = target.cuda()
        adv = adv_generator(args, images, target, model,
                            eps=eps, attack_steps=attack_steps, attack_lr=attack_lr,
                            random_start=random_start, attack_criterion='regular', use_best=True)
        # ensure adv is on CPU for downstream checks if caller wants that
        return adv.cpu()
    else:
        # monkeypatch torch.Tensor.cuda to be identity for the duration of call
        orig_cuda = torch.Tensor.cuda
        try:
            setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)  # no-op
            # keep model, images, target on CPU
            adv = adv_generator(args, images_norm, target, model,
                                eps=eps, attack_steps=attack_steps, attack_lr=attack_lr,
                                random_start=random_start, attack_criterion='regular', use_best=True)
            return adv  # already on CPU
        finally:
            # restore
            setattr(torch.Tensor, "cuda", orig_cuda)


def test_adv_generator_l2_respects_eps_and_shape():
    """
    Run adv_generator with L2 norm and make sure:
    - the returned tensor shape matches input shape
    - per-sample L2 (in denormalized pixel space) <= eps
    """
    # simple dummy model that returns logits on the same device as input
    class DummyModel(nn.Module):
        def __init__(self, num_classes=10,input_dim=3*16*16):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.linear(x.view(x.size(0), -1))

    # args used by adv_generator
    args = SimpleNamespace()
    args.std = [0.229, 0.224, 0.225]
    args.mean = [0.485, 0.456, 0.406]
    args.attack_norm = 'l2'
    args.amp_version = 'none'  # no autocast

    B, C, H, W = 2, 3, 16, 16
    images_norm = torch.rand(B, C, H, W)  # normalized images expected by adv_generator
    target = torch.randint(0, 10, (B,))

    eps = 1.0  # L2 radius in denormalized [0,1] pixel-space
    attack_steps = 3
    attack_lr = 0.25
    random_start = True

    model = DummyModel()
    adv = _run_adv_generator_test(args, images_norm, target, model, eps, attack_steps, attack_lr, random_start)

    # adv is returned normalized -> shape check
    assert adv.shape == (B, C, H, W)

    # compute difference in denormalized space:
    std_tensor = torch.tensor(args.std).view(1, C, 1, 1)
    mean_tensor = torch.tensor(args.mean).view(1, C, 1, 1)

    # original denorm and adv denorm:
    orig_denorm = images_norm * std_tensor + mean_tensor
    adv_denorm = adv * std_tensor + mean_tensor

    norms = per_sample_l2_norm(adv_denorm, orig_denorm)
    assert torch.all(norms <= eps + 1e-4), f"adv_generator L2 produced norms {norms} > {eps}"


def test_adv_generator_linf_respects_eps_and_shape():
    """
    Run adv_generator with Linf norm and make sure:
    - returned tensor shape matches input shape
    - per-sample Linf (in denormalized pixel space) <= eps
    """
    class DummyModel(nn.Module):
        def __init__(self, num_classes=10, input_dim=3*16*16):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.linear(x.view(x.size(0), -1))

    args = SimpleNamespace()
    args.std = [0.229, 0.224, 0.225]
    args.mean = [0.485, 0.456, 0.406]
    args.attack_norm = 'linf'
    args.amp_version = 'none'

    B, C, H, W = 2, 3, 16, 16
    images_norm = torch.rand(B, C, H, W)
    target = torch.randint(0, 10, (B,))

    eps_px = 4.0
    eps = eps_px / 255.0  # adv_generator expects eps in [0,1] denorm pixel space
    attack_steps = 3
    attack_lr = 1.0 / 255.0
    random_start = True

    model = DummyModel()
    adv = _run_adv_generator_test(args, images_norm, target, model, eps, attack_steps, attack_lr, random_start)

    assert adv.shape == (B, C, H, W)

    std_tensor = torch.tensor(args.std).view(1, C, 1, 1)
    mean_tensor = torch.tensor(args.mean).view(1, C, 1, 1)

    orig_denorm = images_norm * std_tensor + mean_tensor
    adv_denorm = adv * std_tensor + mean_tensor

    linf_per_sample = per_sample_linf_norm(adv_denorm, orig_denorm)
    assert torch.all(linf_per_sample <= eps + 1e-6), f"adv_generator Linf produced norms {linf_per_sample} > {eps}"


if __name__ == "__main__":
    # allow running the file directly: python -m pytest tests/test_attacker_steps.py -q
    pytest.main([__file__, "-q"])
