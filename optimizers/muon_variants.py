import math

import torch

from polar import polar
from riemann_aurora import _riemannian_balanced_polar


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
