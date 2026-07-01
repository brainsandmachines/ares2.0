from contextlib import contextmanager, suppress
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''

    def __init__(self, orig_input, eps, step_size, use_grad=True, clamp_min=0.0, clamp_max=1.0):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (torch.Tensor): the original input
        '''

        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def apply_bounds(self, x):
        if self.clamp_min is None or self.clamp_max is None:
            return x
        return torch.clamp(x, self.clamp_min, self.clamp_max)

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            x (torch.Tensor): the input to project back into the feasible set.
        Returns:
            A torch.Tensor that is the input projected back into
            the feasible set, that is, .. math:: \min_{x' \in S} \|x' - x\|_2
        '''

        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p` norms).

        :param x (torch.Tensor): The input to project back into the feasible set.
        :param g (torch.Tensor): The raw gradient
        :return: The new input, a torch.Tensor for the next step.
        '''

        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''

        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''

        return x

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """

    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return self.apply_bounds(diff + self.orig_input)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return self.apply_bounds(new_x)

    def random_uniform(self, x):
        noise=torch.rand_like(x)
        noise.uniform_(-self.eps, self.eps)

        return self.apply_bounds(x + noise)

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        Project the input x back to the ℓ₂ ball around the original input.
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return self.apply_bounds(self.orig_input + diff)

    def step(self, x, g):
        """
        Take a step in the direction of the normalized gradient with step size.
        """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        # Flatten spatial + channel dims per sample
        diff = torch.rand_like(x) - 0.5
        diff = diff.view(diff.size(0), -1)
        diff = diff / (diff.norm(p=2, dim=1, keepdim=True) + 1e-10)
        diff = diff.view_as(x)
        # scale by eps and clamp
        return self.apply_bounds(x + diff * self.eps)
    
    def random_uniform(self, x):
        return self.random_perturb(x)
    

# Assuming AttackerStep is defined elsewhere in your codebase
class L1Step(AttackerStep):
    """
    Attack step for L1 threat model. 
    Projects updates onto the L1 ball: S = {x | ||x - x0||_1 <= epsilon}
    """
    
    def project(self, x):
        """
        Project x back to the L1 ball around orig_input using Duchi's algorithm.
        """
        # 1. Compute perturbation
        diff = x - self.orig_input
        diff_flat = diff.reshape(diff.size(0), -1)
        
        # 2. OPTIMIZATION: Skip projection if already inside L1 ball
        # This saves massive computation time for small perturbations
        current_l1 = torch.norm(diff_flat, p=1, dim=1, keepdim=True)
        if (current_l1 <= self.eps).all():
            return self.apply_bounds(x)

        # 3. Identify violators (batch-wise optimization)
        # We only project samples that violate the constraint
        mask_needs_proj = (current_l1 > self.eps).squeeze()
        if not mask_needs_proj.any():
            return self.apply_bounds(x)
        
        diff_to_proj = diff_flat[mask_needs_proj]
        
        # 4. Robust Duchi Algorithm
        # Use abs values for sorting
        abs_diff = torch.abs(diff_to_proj)
        sorted_vals, _ = torch.sort(abs_diff, dim=1, descending=True)
        
        cumsum = torch.cumsum(sorted_vals, dim=1)
        rho = torch.arange(1, diff_flat.size(1) + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        
        # Calculate candidate thresholds
        theta_candidates = (cumsum - self.eps) / rho
        
        # Find the *last* index where (val - theta) > 0
        # This is the mathematically correct condition for Duchi
        active_cond = (sorted_vals - theta_candidates) > 0
        # Count how many indices satisfy the condition to find the 'cut' point
        num_active = torch.sum(active_cond, dim=1, keepdim=True)
        
        # Gather the valid theta at that index
        # We use (num_active - 1) because indices are 0-based
        idx = torch.clamp(num_active - 1, min=0).long()
        theta_star = torch.gather(theta_candidates, 1, idx)
        
        # 5. Apply Soft-Thresholding
        # sign(v) * max(|v| - theta, 0)
        projected_flat = torch.sign(diff_to_proj) * torch.clamp(abs_diff - theta_star, min=0)
        
        # 6. Scatter back to original tensor
        diff_flat_out = diff_flat.clone()
        diff_flat_out[mask_needs_proj] = projected_flat
        
        # 7. Add back to original input and Box-Constraint [0, 1]
        x_out = self.orig_input + diff_flat_out.reshape_as(diff)
        return self.apply_bounds(x_out)

    def step(self, x, g):
        """
        L1 step mode selector:
        - l2_norm: legacy baseline direction g / ||g||_2
        - l1_apgd: top-k sparse sign updates
        """
        mode = str(getattr(self, "l1_step_mode", "l2_norm")).lower()

        if mode == "l1_apgd":
            g_flat = g.reshape(g.size(0), -1)
            d = g_flat.size(1)
            rho = float(getattr(self, "l1_apgd_rho", 0.05))
            rho = max(0.0, min(1.0, rho))
            k = max(1, int(rho * d))

            _, topk_idx = torch.topk(torch.abs(g_flat), k=k, dim=1, largest=True, sorted=False)
            u_flat = torch.zeros_like(g_flat)
            u_flat.scatter_(1, topk_idx, torch.sign(g_flat.gather(1, topk_idx)))

            u = u_flat.reshape_as(g)
            return x + self.step_size * u

        if mode != "l2_norm":
            raise ValueError(f"Unsupported l1_step_mode: {mode}")

        # Legacy baseline: L2-normalized direction
        g_flat = g.reshape(g.size(0), -1)
        l2_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True).reshape(-1, 1, 1, 1)
        grad_normalized = g / (l2_norm + 1e-10)
        return x + grad_normalized * self.step_size

    def random_perturb(self, x):
        """
        Random perturbation.
        For L1, sparse noise is often better, but uniform L1 noise is acceptable for restart.
        """
        # Generate random noise
        diff = torch.rand_like(x) - 0.5
        diff = diff.reshape(diff.size(0), -1)

        # Normalize to have L1 norm = 1, then scale by epsilon
        # Note: This puts points on the surface of the L1 ball
        norm = torch.norm(diff, p=1, dim=1, keepdim=True)
        diff = diff / (norm + 1e-10)
        diff = diff.reshape_as(x) * self.eps
        
        return self.apply_bounds(x + diff)

    def random_uniform(self, x):
        return self.random_perturb(x)
    
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def replace_best(loss, bloss, x, bx):
    if bloss is None:
        bx = x.clone().detach()
        bloss = loss.clone().detach()
    else:
        replace = bloss < loss
        bx[replace] = x[replace].clone().detach()
        bloss[replace] = loss[replace]

    return bloss, bx

def replace_best_reverse(loss, bloss, x, bx):
    if bloss is None:
        bx = x.clone().detach()
        bloss = loss.clone().detach()
    else:
        replace = bloss > loss
        bx[replace] = x[replace].clone().detach()
        bloss[replace] = loss[replace]

    return bloss, bx


def _build_step(cfg, orig_input, eps, attack_lr, clamp_min=0.0, clamp_max=1.0):
    attacks = cfg.attacks
    if attacks.attack_norm == 'l2':
        return L2Step(eps=eps, orig_input=orig_input, step_size=attack_lr, clamp_min=clamp_min, clamp_max=clamp_max)
    if attacks.attack_norm == 'l1':
        step = L1Step(eps=eps, orig_input=orig_input, step_size=attack_lr, clamp_min=clamp_min, clamp_max=clamp_max)
        step.l1_step_mode = str(attacks.get('l1_step_mode', 'l2_norm')).lower()
        step.l1_apgd_rho = float(attacks.get('l1_apgd_rho', 0.05))
        return step
    if attacks.attack_norm == 'linf':
        return LinfStep(eps=eps, orig_input=orig_input, step_size=attack_lr, clamp_min=clamp_min, clamp_max=clamp_max)
    raise NotImplementedError


def _configure_l1_step_search(cfg, attack_lr):
    attacks = cfg.attacks
    return (
        bool(attacks.get('l1_apgd_use_halving', True)),
        bool(attacks.attack_norm == 'l1' and str(attacks.get('l1_step_mode', 'l2_norm')).lower() == 'l1_apgd'),
        float(attack_lr),
        max(float(attack_lr) * float(attacks.get('l1_apgd_min_step_scale', 0.01)), 1e-12),
        None,
    )


def _unwrap_v1_attack_model(model):
    base_model = model.module if hasattr(model, 'module') else model
    required = ('forward_v1_features', 'forward_from_v1_features', 'forward_with_v1_override')
    if not all(hasattr(base_model, attr) for attr in required):
        raise ValueError('V1 feature-space attacks require a model with the V1 attack interface.')
    return base_model


@contextmanager
def _temporarily_disable_param_grads(model):
    params = list(model.parameters())
    previous = [param.requires_grad for param in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, requires_grad in zip(params, previous):
            param.requires_grad_(requires_grad)

def adv_generator(cfg, images, target, model, eps, attack_steps, attack_lr, random_start, use_best=True):
    # denorm images to 0-1
    std_tensor=torch.Tensor(cfg.dataset.std).cuda(non_blocking=True)[None, :, None, None]
    mean_tensor=torch.Tensor(cfg.dataset.mean).cuda(non_blocking=True)[None, :, None, None]
    images=images*std_tensor+mean_tensor

    # define perturbation range
    prev_training = bool(model.training)
    model.eval()
    try:
        with _temporarily_disable_param_grads(model):
            orig_input = images.detach().cuda(non_blocking=True)
            step = _build_step(cfg, orig_input, eps, attack_lr)

            # define loss function
            attack_criterion = torch.nn.CrossEntropyLoss()

            # amp_autocast
            amp_autocast=suppress
            if cfg.amp_version == 'native':
                amp_autocast = torch.cuda.amp.autocast

            # adv generation
            best_loss = None
            best_x = None
            if random_start:
                images = step.random_perturb(images)
            else:
                images = orig_input.clone().detach()

            l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = _configure_l1_step_search(cfg, attack_lr)

            for _ in range(attack_steps):
                images = images.clone().detach().requires_grad_(True)

                # forward
                adv_losses=None
                with amp_autocast():
                    adv_losses = attack_criterion(model((images-mean_tensor)/std_tensor), target)

                # update gradient
                grad = torch.autograd.grad(torch.mean(adv_losses), images)[0].detach()
                with torch.no_grad():
                    if l1_apgd_mode_active:
                        cur_loss = float(adv_losses.item())
                        if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                            l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                        step.step_size = l1_cur_step
                        l1_prev_loss = cur_loss

                    varlist = [adv_losses, best_loss, images, best_x]
                    best_loss, best_x = replace_best(*varlist) if use_best else (adv_losses, images)

                    images = step.step(images, grad)
                    images = step.project(images)
        with torch.no_grad():
            adv_losses = attack_criterion(model((images-mean_tensor)/std_tensor), target)
        varlist = [adv_losses, best_loss, images, best_x]
        best_loss, best_x = replace_best(*varlist) if use_best else (adv_losses, images)
    finally:
        if prev_training:
            model.train()
    
    best_x=(best_x-mean_tensor)/std_tensor
    return best_x


def v1_adv_generator(cfg, images, target, model, eps, attack_steps, attack_lr, random_start, use_best=True):
    prev_training = bool(model.training)
    model.eval()
    attack_model = _unwrap_v1_attack_model(model)
    try:
        with _temporarily_disable_param_grads(model):
            orig_features = attack_model.forward_v1_features(images, apply_noise=True).detach()
            step = _build_step(cfg, orig_features, eps, attack_lr, clamp_min=None, clamp_max=None)
            attack_criterion = torch.nn.CrossEntropyLoss()

            amp_autocast = suppress
            if cfg.amp_version == 'native':
                amp_autocast = torch.cuda.amp.autocast

            best_loss = None
            best_x = None
            if random_start:
                adv_features = step.random_perturb(orig_features)
            else:
                adv_features = orig_features.clone().detach()

            l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = _configure_l1_step_search(cfg, attack_lr)

            for _ in range(attack_steps):
                adv_features = adv_features.clone().detach().requires_grad_(True)
                with amp_autocast():
                    adv_losses = attack_criterion(attack_model.forward_from_v1_features(adv_features), target)

                grad = torch.autograd.grad(torch.mean(adv_losses), adv_features)[0].detach()
                with torch.no_grad():
                    if l1_apgd_mode_active:
                        cur_loss = float(adv_losses.item())
                        if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                            l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                        step.step_size = l1_cur_step
                        l1_prev_loss = cur_loss

                    best_loss, best_x = replace_best(adv_losses, best_loss, adv_features, best_x) if use_best else (adv_losses, adv_features)
                    adv_features = step.step(adv_features, grad)
                    adv_features = step.project(adv_features)

            with torch.no_grad():
                adv_losses = attack_criterion(attack_model.forward_from_v1_features(adv_features), target)
            best_loss, best_x = replace_best(adv_losses, best_loss, adv_features, best_x) if use_best else (adv_losses, adv_features)
    finally:
        if prev_training:
            model.train()
    return best_x.detach()

def v1_trades_adv_generator(cfg, images, model, eps, attack_steps, attack_lr, random_start=True, use_best=True):
    prev_training = bool(model.training)
    model.eval()
    attack_model = _unwrap_v1_attack_model(model)

    with torch.no_grad():
        nat_features = attack_model.forward_v1_features(images, apply_noise=False).detach()
        nat_logits = attack_model.forward_from_v1_features(nat_features)
        nat_probs = torch.softmax(nat_logits, dim=1).detach()

    step = _build_step(cfg, nat_features, eps, attack_lr, clamp_min=None, clamp_max=None)

    if random_start:
        adv_features = step.random_perturb(nat_features)
    else:
        adv_features = nat_features.clone()

    best_loss = None
    best_x = None

    l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = _configure_l1_step_search(cfg, attack_lr)

    for _ in range(attack_steps):
        adv_features = adv_features.clone().detach().requires_grad_(True)
        logits_adv = attack_model.forward_from_v1_features(adv_features)

        kl = torch.nn.functional.kl_div(
            torch.log_softmax(logits_adv, dim=1),
            nat_probs,
            reduction='none'
        ).sum(dim=1)

        grad = torch.autograd.grad(kl.sum(), adv_features)[0]

        with torch.no_grad():
            if l1_apgd_mode_active:
                cur_loss = float(kl.mean().item())
                if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                    l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                step.step_size = l1_cur_step
                l1_prev_loss = cur_loss

            best_loss, best_x = replace_best(kl, best_loss, adv_features, best_x) if use_best else (kl, adv_features)
            adv_features = step.step(adv_features, grad)
            adv_features = step.project(adv_features)

    with torch.no_grad():
        logits_adv = attack_model.forward_from_v1_features(adv_features)
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(logits_adv, dim=1),
            nat_probs,
            reduction='none'
        ).sum(dim=1)
        best_loss, best_x = replace_best(kl.detach(), best_loss, adv_features.detach(), best_x) if use_best else (kl, adv_features.detach())

    if prev_training:
        model.train()
    return best_x.detach()

def trades_adv_generator(cfg, images, model, eps, attack_steps, attack_lr, random_start=True, use_best=True):
    prev_training = bool(model.training)
    std = torch.Tensor(cfg.dataset.std).cuda()[None,:,None,None]
    mean = torch.Tensor(cfg.dataset.mean).cuda()[None,:,None,None]
    images = images * std + mean

    model.eval()
    x_nat = images.detach()

    with torch.no_grad():
        nat_logits = model((x_nat - mean) / std)
        nat_probs = torch.softmax(nat_logits, dim=1).detach()

    step = _build_step(cfg, x_nat, eps, attack_lr)

    if random_start:
        x_adv = step.random_perturb(x_nat)
    else:
        x_adv = x_nat.clone()

    best_loss = None
    best_x = None

    l1_apgd_use_halving, l1_apgd_mode_active, l1_cur_step, l1_min_step, l1_prev_loss = _configure_l1_step_search(cfg, attack_lr)

    for _ in range(attack_steps):
        x_adv.requires_grad_()
        logits_adv = model((x_adv - mean) / std)

        kl = torch.nn.functional.kl_div(
            torch.log_softmax(logits_adv, dim=1),
            nat_probs,
            reduction='none'
        ).sum(dim=1)   # per-sample KL

        grad = torch.autograd.grad(kl.sum(), x_adv)[0]

        with torch.no_grad():
            if l1_apgd_mode_active:
                cur_loss = float(kl.mean().item())
                if l1_apgd_use_halving and l1_prev_loss is not None and cur_loss <= l1_prev_loss + 1e-12:
                    l1_cur_step = max(l1_cur_step / 2.0, l1_min_step)
                step.step_size = l1_cur_step
                l1_prev_loss = cur_loss

            best_loss, best_x = replace_best(kl, best_loss, x_adv, best_x) if use_best else (kl, x_adv)
            x_adv = step.step(x_adv, grad)
            x_adv = step.project(x_adv)

    # Final selection safety
    with torch.no_grad():
        logits_adv = model((x_adv - mean) / std)
        kl = torch.nn.functional.kl_div(
        torch.log_softmax(logits_adv, dim=1),
        nat_probs,
        reduction='none'
    ).sum(dim=1)
        best_loss, best_x = replace_best(kl.detach(), best_loss, x_adv.detach(), best_x) if use_best else (kl, x_adv.detach())
    if prev_training:
        model.train()
    best_x = torch.clamp(best_x, 0.0, 1.0)
    return (best_x - mean) / std

class LinfStepEps(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """

    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.minimum(torch.maximum(diff, -self.eps.repeat(x.size()[1],x.size()[2],x.size()[3],1).permute(3,0,1,2)), self.eps.repeat(x.size()[1],x.size()[2],x.size()[3],1).permute(3,0,1,2))
        return torch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size.repeat(x.size()[1],x.size()[2],x.size()[3],1).permute(3,0,1,2)
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps.repeat(x.size()[1],x.size()[2],x.size()[3],1).permute(3,0,1,2)
        return torch.clamp(new_x, 0, 1)

class PgdAttackEps(object):
    def __init__(self, model, batch_size, goal, distance_metric, magnitude=4/255, alpha=1/255, iteration=100, num_restart=1, random_start=True, use_best=False, device=None, _logger=None):
        self.model=model
        self.batch_size=batch_size
        self.goal=goal
        self.distance_metric=distance_metric
        self.magnitude=magnitude
        self.alpha=alpha
        self.iteration=iteration
        self.num_restart=num_restart
        self.random_start=random_start
        self.use_best=use_best
        self.device=device
        self._logger=_logger

    def config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def batch_attack(self, input_image, label, target_label):
        # generate adversarial examples
        prev_training = bool(self.model.training)
        self.model.eval()
        orig_input = input_image.clone().detach().cuda(non_blocking=True)

        attack_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        best_loss = None
        best_x = None

        for _ in range(self.num_restart):
            step = LinfStepEps(eps=self.magnitude, orig_input=orig_input, step_size=self.alpha)
            images = orig_input.clone().detach()

            if self.random_start:
                images = step.random_perturb(images) 
            for _ in range(self.iteration):
                images = images.clone().detach().requires_grad_(True)
                adv_losses = attack_criterion(self.model(images), label)
                torch.mean(adv_losses).backward()
                grad = images.grad.detach()

                with torch.no_grad():
                    varlist = [adv_losses, best_loss, images, best_x]
                    best_loss, best_x = replace_best(*varlist) if self.use_best else (adv_losses, images)

                    images = step.step(images, grad)
                    images = step.project(images)

            adv_losses = attack_criterion(self.model(images), label)
            varlist = [adv_losses, best_loss, images, best_x]
            best_loss, best_x = replace_best(*varlist) if self.use_best else (adv_losses, images)
        if prev_training:
            self.model.train()
    
        return best_x
    

 
