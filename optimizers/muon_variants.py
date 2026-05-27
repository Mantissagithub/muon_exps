import math

import torch

from polar import polar

try:
    from riemann_aurora import _riemannian_balanced_polar
except ModuleNotFoundError:
    _riemannian_balanced_polar = None


class _MatrixOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        mu=0.95,
        nesterov=True,
        eps=1e-7,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            nesterov=nesterov,
            eps=eps,
        )
        super().__init__(params, defaults)

    def _momentum_update(self, p: torch.Tensor, grad: torch.Tensor, group: dict) -> torch.Tensor:
        if p.ndim != 2:
            raise ValueError(f"{self.__class__.__name__} only supports 2D tensors, got {tuple(p.shape)}")

        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p)

        mu = group["mu"]
        momentum = state["momentum"]
        momentum.lerp_(grad, 1.0 - mu)
        if group["nesterov"]:
            return torch.lerp(grad, momentum, mu)
        return momentum.clone()

    def _apply_update(self, p: torch.Tensor, update: torch.Tensor, lr: float, weight_decay: float) -> None:
        p.mul_(1.0 - lr * weight_decay)
        p.add_(update, alpha=-lr)


class MuonLike(_MatrixOptimizer):
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = self._momentum_update(p, p.grad, group)
                update = polar(update, eps=group["eps"])
                scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                update.mul_(scale)
                self._apply_update(p, update, group["lr"], group["weight_decay"])
        return loss


class NorMuon(_MatrixOptimizer):
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = self._momentum_update(p, p.grad, group)
                update = polar(update, eps=group["eps"])

                state = self.state[p]
                if "row_ema" not in state:
                    state["row_ema"] = torch.ones(p.size(0), 1, device=p.device, dtype=torch.float32)
                row_ema = state["row_ema"]
                row_mean_sq = update.to(torch.float32).pow(2).mean(dim=1, keepdim=True)
                row_ema.mul_(0.999).add_(row_mean_sq, alpha=0.001)
                update = update / (row_ema.to(update.dtype).sqrt() + group["eps"])

                lr_hat = 0.2 * group["lr"] * update.norm().item() / math.sqrt(update.numel())
                self._apply_update(p, update, lr_hat, group["weight_decay"])
        return loss


class UNorMuon(_MatrixOptimizer):
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = self._momentum_update(p, p.grad, group)
                update = polar(update, eps=group["eps"])

                state = self.state[p]
                if "row_ema" not in state:
                    state["row_ema"] = torch.ones(p.size(0), 1, device=p.device, dtype=torch.float32)
                row_ema = state["row_ema"]
                row_mean_sq = update.to(torch.float32).pow(2).mean(dim=1, keepdim=True)
                row_ema.mul_(0.999).add_(row_mean_sq, alpha=0.001)
                update = update / (row_ema.to(update.dtype).sqrt() + group["eps"])

                lr_hat = 0.2 * group["lr"] * update.norm().item() / math.sqrt(p.size(1))
                self._apply_update(p, update, lr_hat, group["weight_decay"])
        return loss


class Aurora(_MatrixOptimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, mu=0.95, nesterov=True, pp_iterations=2, pp_beta=0.5, eps=1e-7):
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu, nesterov=nesterov, eps=eps)
        for group in self.param_groups:
            group["pp_iterations"] = pp_iterations
            group["pp_beta"] = pp_beta

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = self._momentum_update(p, p.grad, group)
                m, n = update.shape
                if m == n:
                    update = polar(update, eps=group["eps"])
                else:
                    transposed = m < n
                    work = update.mT if transposed else update
                    work = work.to(torch.float32)
                    rows, cols = work.shape
                    target_row_sq = cols / rows
                    row_norm = work.norm(dim=1, keepdim=True).clamp_min(group["eps"])
                    d = 1.0 / row_norm
                    for k in range(group["pp_iterations"]):
                        u = polar(d * work, eps=group["eps"])
                        if k < group["pp_iterations"] - 1:
                            row_sq = u.to(torch.float32).pow(2).sum(dim=1, keepdim=True).clamp_min(group["eps"] ** 2)
                            d = d * (target_row_sq / row_sq).pow(group["pp_beta"])
                    update = u.mT if transposed else u
                    update = update.to(dtype=p.dtype)
                update.mul_(math.sqrt(max(1.0, p.size(0) / p.size(1))))
                self._apply_update(p, update, group["lr"], group["weight_decay"])
        return loss


class RiemannAurora(_MatrixOptimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        mu=0.95,
        nesterov=True,
        outer_steps=2,
        cg_steps=8,
        riemannian_eta=0.1,
        retraction_steps=1,
        eps=1e-7,
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu, nesterov=nesterov, eps=eps)
        for group in self.param_groups:
            group["outer_steps"] = outer_steps
            group["cg_steps"] = cg_steps
            group["riemannian_eta"] = riemannian_eta
            group["retraction_steps"] = retraction_steps

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = self._momentum_update(p, p.grad, group)
                if update.size(0) == update.size(1):
                    update = polar(update, eps=group["eps"])
                else:
                    if _riemannian_balanced_polar is None:
                        raise RuntimeError("riemann_aurora.py is required for non-square RiemannAurora updates")
                    update = _riemannian_balanced_polar(
                        update,
                        outer_steps=group["outer_steps"],
                        cg_steps=group["cg_steps"],
                        riemannian_eta=group["riemannian_eta"],
                        retraction_steps=group["retraction_steps"],
                        eps=group["eps"],
                    )
                update.mul_(math.sqrt(max(1.0, p.size(0) / p.size(1))))
                self._apply_update(p, update.to(dtype=p.dtype), group["lr"], group["weight_decay"])
        return loss


class AMUSEMuon:
    """Schedule-free Muon wrapper for the TinyShakespeare benchmark.

    Model parameters hold y_t during training, while this object keeps explicit
    z_t and x_t copies. That is clearer for the first implementation, but it
    costs two full parameter copies; a lower-memory version can reconstruct x_t
    from y_t/z_t and store only one sequence plus optimizer state.
    """

    def __init__(
        self,
        matrix_params,
        other_params,
        lr=1e-3,
        weight_decay=0.1,
        mu=0.95,
        nesterov=True,
        eps=1e-7,
        beta1=0.6,
        rho=0.8,
        warmup_steps=100,
        fixed_beta=None,
        fallback_betas=(0.9, 0.95),
        fallback_eps=1e-8,
    ):
        self.matrix_params = list(matrix_params)
        self.other_params = list(other_params)
        self.params = self.matrix_params + self.other_params
        self.matrix_ids = {id(p) for p in self.matrix_params}
        self.lr = lr
        self.weight_decay = weight_decay
        self.mu = mu
        self.nesterov = nesterov
        self.eps = eps
        self.beta1 = beta1
        self.rho = rho
        self.warmup_steps = max(1, int(warmup_steps))
        self.fixed_beta = fixed_beta
        self.fallback_beta1, self.fallback_beta2 = fallback_betas
        self.fallback_eps = fallback_eps
        self.step_idx = 0
        self.last_beta = self.beta_t(0)
        self.last_update_cosine = float("nan")
        self._prev_delta_x = None

        self.state = {}
        for p in self.params:
            self.state[p] = {
                "z": p.detach().clone(),
                "x": p.detach().clone(),
            }
            if id(p) in self.matrix_ids:
                self.state[p]["momentum"] = torch.zeros_like(p)
            else:
                self.state[p]["exp_avg"] = torch.zeros_like(p)
                self.state[p]["exp_avg_sq"] = torch.zeros_like(p)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None if set_to_none else torch.zeros_like(p)

    def beta_t(self, step_idx):
        if self.fixed_beta is not None:
            return float(self.fixed_beta)
        t = step_idx + 1
        if t <= self.warmup_steps:
            return float(self.beta1)
        beta = 1.0 - ((self.warmup_steps - 1.0) / max(t - 1.0, 1.0)) ** self.rho * (1.0 - self.beta1)
        return float(min(max(beta, self.beta1), 1.0))

    @torch.no_grad()
    def prepare_train_step(self):
        beta = self.beta_t(self.step_idx)
        self.last_beta = beta
        for p in self.params:
            state = self.state[p]
            p.copy_(state["z"])
            p.lerp_(state["x"], beta)

    @torch.no_grad()
    def use_x_params(self):
        for p in self.params:
            p.copy_(self.state[p]["x"])

    @torch.no_grad()
    def use_z_params(self):
        for p in self.params:
            p.copy_(self.state[p]["z"])

    @torch.no_grad()
    def step(self):
        delta_chunks = []
        for p in self.matrix_params:
            if p.grad is None:
                continue
            state = self.state[p]
            z = state["z"]
            grad = p.grad

            z.mul_(1.0 - self.lr * self.weight_decay)
            momentum = state["momentum"]
            momentum.lerp_(grad, 1.0 - self.mu)
            update = torch.lerp(grad, momentum, self.mu) if self.nesterov else momentum
            update = polar(update, eps=self.eps)
            update.mul_(math.sqrt(max(1.0, z.size(0) / z.size(1))))
            z.add_(update, alpha=-self.lr)
            delta_chunks.append(self._update_x(state, z))

        for p in self.other_params:
            if p.grad is None:
                continue
            state = self.state[p]
            z = state["z"]
            grad = p.grad

            z.mul_(1.0 - self.lr * self.weight_decay)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.fallback_beta1).add_(grad, alpha=1.0 - self.fallback_beta1)
            exp_avg_sq.mul_(self.fallback_beta2).addcmul_(grad, grad, value=1.0 - self.fallback_beta2)

            t = self.step_idx + 1
            bias_correction1 = 1.0 - self.fallback_beta1**t
            bias_correction2 = 1.0 - self.fallback_beta2**t
            step_size = self.lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.fallback_eps)
            z.addcdiv_(exp_avg, denom, value=-step_size)
            delta_chunks.append(self._update_x(state, z))

        self.last_update_cosine = self._update_cosine(delta_chunks)
        self.step_idx += 1
        self.prepare_train_step()

    def _update_x(self, state, z):
        x = state["x"]
        old_x = x.detach().clone()
        c = 1.0 / (self.step_idx + 1.0)
        x.lerp_(z, c)
        return (x - old_x).detach().flatten().to(torch.float32)

    def _update_cosine(self, delta_chunks):
        if not delta_chunks:
            return float("nan")
        delta = torch.cat(delta_chunks)
        if self._prev_delta_x is None:
            self._prev_delta_x = delta.clone()
            return float("nan")
        denom = delta.norm() * self._prev_delta_x.norm()
        cosine = float("nan") if denom.item() == 0.0 else torch.dot(delta, self._prev_delta_x).div(denom).item()
        self._prev_delta_x = delta.clone()
        return cosine

    @torch.no_grad()
    def metrics(self):
        zx_num = zx_den = yx_num = yx_den = yz_num = yz_den = 0.0
        beta = self.last_beta
        for p in self.params:
            state = self.state[p]
            z = state["z"].to(torch.float32)
            x = state["x"].to(torch.float32)
            y = torch.lerp(z, x, beta)
            zx_num += torch.sum((z - x).pow(2)).item()
            zx_den += torch.sum(x.pow(2)).item()
            yx_num += torch.sum((y - x).pow(2)).item()
            yx_den += torch.sum(x.pow(2)).item()
            yz_num += torch.sum((y - z).pow(2)).item()
            yz_den += torch.sum(z.pow(2)).item()
        return {
            "beta_t": beta,
            "update_cosine_similarity": self.last_update_cosine,
            "z_x_distance": math.sqrt(zx_num) / max(math.sqrt(zx_den), 1e-12),
            "y_x_distance": math.sqrt(yx_num) / max(math.sqrt(yx_den), 1e-12),
            "y_z_distance": math.sqrt(yz_num) / max(math.sqrt(yz_den), 1e-12),
        }
