import math

import torch

from polar import MODDED_NANOGPT_NS_COEFFS, polar

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


def _aurora_balanced_polar(
    update: torch.Tensor,
    eps: float,
    pp_iterations: int = 1,
    pp_beta: float = 0.5,
    polar_steps: int = 5,
    polar_dtype: torch.dtype | None = None,
    polar_coeffs: tuple[tuple[float, float, float], ...] | None = None,
) -> torch.Tensor:
    m, n = update.shape
    if m == n:
        return polar(update, steps=polar_steps, eps=eps, compute_dtype=polar_dtype, coeffs=polar_coeffs)

    transposed = m < n
    work = update.mT if transposed else update
    work = work.to(torch.float32)
    rows, cols = work.shape
    target_row_sq = cols / rows
    row_norm = work.norm(dim=1, keepdim=True).clamp_min(eps)
    d = 1.0 / row_norm
    u = None
    for k in range(pp_iterations):
        u = polar(d * work, steps=polar_steps, eps=eps, compute_dtype=polar_dtype, coeffs=polar_coeffs)
        if k < pp_iterations - 1:
            row_sq = u.to(torch.float32).pow(2).sum(dim=1, keepdim=True).clamp_min(eps**2)
            d = d * (target_row_sq / row_sq).pow(pp_beta)
    if u is None:
        u = polar(work, steps=polar_steps, eps=eps, compute_dtype=polar_dtype, coeffs=polar_coeffs)
    return u.mT if transposed else u


def _as_named_entries(named_params):
    entries = list(named_params)
    entries.sort(key=lambda item: item[0])
    return entries


def _relative_norm(numer_sq, denom_sq, eps):
    return math.sqrt(numer_sq) / max(math.sqrt(denom_sq), eps)


class AMUSEMuon:
    """Schedule-free Muon wrapper for the TinyShakespeare benchmark.

    Model parameters hold y_t during training, while this object keeps explicit
    z_t and x_t copies. That is clearer for the first implementation, but it
    costs two full parameter copies; a lower-memory version can reconstruct x_t
    from y_t/z_t and store only one sequence plus optimizer state.
    """

    def __init__(
        self,
        matrix_named_params,
        other_named_params,
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
        debug_checks=False,
        track_update_cosine=True,
        factored_second_moment=False,
        polar_steps=5,
        polar_dtype=None,
        polar_coeffs=None,
    ):
        self.matrix_entries = _as_named_entries(matrix_named_params)
        self.other_entries = _as_named_entries(other_named_params)
        self.param_entries = self.matrix_entries + self.other_entries
        self.matrix_ids = {id(param) for _, param in self.matrix_entries}
        self.params = [param for _, param in self.param_entries]
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
        self.debug_checks = debug_checks
        self.track_update_cosine = track_update_cosine
        self.factored_second_moment = factored_second_moment
        self.polar_steps = polar_steps
        self.polar_dtype = polar_dtype
        self.polar_coeffs = polar_coeffs
        self.step_idx = 0
        self.c_t = 0.0
        self.c_t0 = None
        self.last_beta = self.beta_t()
        self.last_update_cosine = float("nan")
        self.last_reconstruction_error = float("nan")
        self.last_raw_cosine = float("nan")
        self._x_prev = None
        self._x_prev_prev = None

        self.state = {}
        for name, p in self.param_entries:
            self.state[p] = {
                "name": name,
                "z": p.detach().clone(),
                "x": p.detach().clone(),
            }
            if id(p) in self.matrix_ids:
                if self.factored_second_moment:
                    self.state[p]["exp_avg_sq_row"] = torch.zeros(p.size(0), 1, device=p.device, dtype=p.dtype)
                    self.state[p]["exp_avg_sq_col"] = torch.zeros(1, p.size(1), device=p.device, dtype=p.dtype)
                else:
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p)
            else:
                self.state[p]["exp_avg"] = torch.zeros_like(p)
                self.state[p]["exp_avg_sq"] = torch.zeros_like(p)
        if self.track_update_cosine:
            self._x_prev = self._flatten_x_snapshot()

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None if set_to_none else torch.zeros_like(p)

    def beta_t(self):
        if self.fixed_beta is not None:
            return float(self.fixed_beta)
        if self.step_idx <= self.warmup_steps or self.c_t0 is None or self.c_t <= self.c_t0:
            return float(self.beta1)
        # after warmup we grow beta from c_t0, not from raw step count
        beta = 1.0 - (self.c_t0 / self.c_t) ** self.rho * (1.0 - self.beta1)
        return float(min(max(beta, self.beta1), 1.0))

    @torch.no_grad()
    def prepare_train_step(self):
        beta = self.beta_t()
        self.last_beta = beta
        for _, p in self.param_entries:
            state = self.state[p]
            p.copy_(state["z"])
            p.lerp_(state["x"], beta)

    @torch.no_grad()
    def use_x_params(self):
        for _, p in self.param_entries:
            p.copy_(self.state[p]["x"])

    @torch.no_grad()
    def use_z_params(self):
        for _, p in self.param_entries:
            p.copy_(self.state[p]["z"])

    @torch.no_grad()
    def use_y_params(self):
        self.prepare_train_step()

    @torch.no_grad()
    def step(self):
        lr_sq = self.lr * self.lr
        # x_t is the lr^2-weighted average of z_t updates
        next_c_t = self.c_t + lr_sq
        x_weight = 0.0 if next_c_t == 0.0 else lr_sq / next_c_t

        for _, p in self.matrix_entries:
            if p.grad is None:
                continue
            state = self.state[p]
            z = state["z"]
            grad = p.grad

            z.mul_(1.0 - self.lr * self.weight_decay)
            t = self.step_idx + 1
            bias_correction2 = 1.0 - self.fallback_beta2**t
            if self.factored_second_moment:
                grad_sq = grad.square()
                row = state["exp_avg_sq_row"]
                col = state["exp_avg_sq_col"]
                row.mul_(self.fallback_beta2).add_(grad_sq.mean(dim=1, keepdim=True), alpha=1.0 - self.fallback_beta2)
                col.mul_(self.fallback_beta2).add_(grad_sq.mean(dim=0, keepdim=True), alpha=1.0 - self.fallback_beta2)
                preconditioned = grad.mul((bias_correction2 * row.mean().clamp_min(self.fallback_eps)).sqrt())
                preconditioned.div_(row.sqrt().clamp_min(self.fallback_eps))
                preconditioned.div_(col.sqrt().clamp_min(self.fallback_eps))
            else:
                exp_avg_sq = state["exp_avg_sq"]
                # amuse matrix path does not use adam first moment, only the second moment
                exp_avg_sq.mul_(self.fallback_beta2).addcmul_(grad, grad, value=1.0 - self.fallback_beta2)
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.fallback_eps)
                preconditioned = grad / denom
            update = self._matrix_update(preconditioned, z)
            # match the preconditioned grad rms before the muon step, same idea as d-muon
            update.mul_(preconditioned.norm() / update.norm().clamp_min(self.eps))
            z.add_(update, alpha=-self.lr)
            self._update_x(state, z, x_weight)

        for _, p in self.other_entries:
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
            self._update_x(state, z, x_weight)

        self.c_t = next_c_t
        if self.step_idx + 1 == self.warmup_steps and self.c_t0 is None:
            # this is the c_T0 anchor we keep for beta growth after warmup
            self.c_t0 = self.c_t

        if self.track_update_cosine:
            self.last_update_cosine = self._update_cosine()
        self.step_idx += 1
        self.prepare_train_step()

    def _update_x(self, state, z, x_weight):
        x = state["x"]
        x.lerp_(z, x_weight)

    def _matrix_update(self, preconditioned_grad, z):
        update = polar(
            preconditioned_grad,
            steps=self.polar_steps,
            eps=self.eps,
            compute_dtype=self.polar_dtype,
            coeffs=self.polar_coeffs,
        )
        update.mul_(math.sqrt(max(1.0, z.size(0) / z.size(1))))
        return update

    def _flatten_x_snapshot(self):
        return torch.cat(
            [self.state[param]["x"].detach().double().flatten().cpu() for _, param in self.param_entries]
        )

    def _update_cosine(self):
        x_now = self._flatten_x_snapshot()
        if self._x_prev is None:
            self._x_prev = x_now
            return float("nan")
        if self._x_prev_prev is None:
            self._x_prev_prev = self._x_prev
            self._x_prev = x_now
            return float("nan")

        dx = x_now - self._x_prev
        dx_prev = self._x_prev - self._x_prev_prev
        denom = dx.norm() * dx_prev.norm()
        if denom.item() < 1e-20:
            raw_cosine = float("nan")
            cosine = float("nan")
        else:
            raw_cosine = torch.dot(dx, dx_prev).div(denom).item()
            if self.debug_checks and (raw_cosine > 1.0001 or raw_cosine < -1.0001):
                print(
                    f"[amuse warning] step={self.step_idx} raw_cosine={raw_cosine:.8f} "
                    f"dx_norm={dx.norm().item():.3e} prev_dx_norm={dx_prev.norm().item():.3e}"
                )
            cosine = float(torch.clamp(torch.tensor(raw_cosine, dtype=torch.float64), -1.0, 1.0).item())

        self.last_raw_cosine = raw_cosine
        self._x_prev_prev = self._x_prev
        self._x_prev = x_now
        return cosine

    @torch.no_grad()
    def metrics(self):
        zx_num = zx_den = yx_num = yx_den = yz_num = yz_den = 0.0
        reconstruction_num = reconstruction_den = 0.0
        beta = self.last_beta
        for _, p in self.param_entries:
            state = self.state[p]
            z = state["z"].to(torch.float64)
            x = state["x"].to(torch.float64)
            y = torch.lerp(z, x, beta)
            model_y = p.detach().double()
            zx_delta = z - x
            yx_delta = y - x
            yz_delta = y - z
            zx_num += torch.sum(zx_delta.pow(2)).item()
            zx_den += torch.sum(x.pow(2)).item()
            yx_num += torch.sum(yx_delta.pow(2)).item()
            yx_den += torch.sum(x.pow(2)).item()
            yz_num += torch.sum(yz_delta.pow(2)).item()
            yz_den += torch.sum(z.pow(2)).item()
            reconstruction_num += torch.sum((model_y - y).pow(2)).item()
            reconstruction_den += torch.sum(model_y.pow(2)).item()

        zx_distance = _relative_norm(zx_num, zx_den, 1e-12)
        yx_distance = _relative_norm(yx_num, yx_den, 1e-12)
        yz_distance = _relative_norm(yz_num, yz_den, 1e-12)
        self.last_reconstruction_error = _relative_norm(reconstruction_num, reconstruction_den, 1e-12)
        if self.debug_checks and self.last_reconstruction_error > 1e-8:
            print(f"[amuse warning] step={self.step_idx} reconstruction_error={self.last_reconstruction_error:.3e}")
        return {
            "beta_t": beta,
            "update_cosine_similarity": self.last_update_cosine,
            "z_x_distance": zx_distance,
            "y_x_distance": yx_distance,
            "y_z_distance": yz_distance,
            "y_x_expected_distance": (1.0 - beta) * zx_distance,
            "y_z_expected_distance": beta * _relative_norm(zx_num, yz_den, 1e-12),
            "amuse_y_reconstruction_error": self.last_reconstruction_error,
        }


class AMUSEAurora(AMUSEMuon):
    """AMUSE schedule-free state with Aurora row-balanced matrix updates."""

    def __init__(self, *args, pp_iterations=1, pp_beta=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.pp_iterations = pp_iterations
        self.pp_beta = pp_beta

    def _matrix_update(self, preconditioned_grad, z):
        update = _aurora_balanced_polar(
            preconditioned_grad,
            eps=self.eps,
            pp_iterations=self.pp_iterations,
            pp_beta=self.pp_beta,
            polar_steps=self.polar_steps,
            polar_dtype=self.polar_dtype,
            polar_coeffs=self.polar_coeffs,
        )
        update.mul_(math.sqrt(max(1.0, z.size(0) / z.size(1))))
        return update.to(dtype=z.dtype)


class AMUSEAuroraFast(AMUSEAurora):
    """Lower-bandwidth AMUSE-Aurora with factored matrix second moments."""

    def __init__(self, *args, polar_steps=4, polar_dtype=torch.float16, **kwargs):
        kwargs.setdefault("factored_second_moment", True)
        kwargs.setdefault("track_update_cosine", False)
        super().__init__(*args, polar_steps=polar_steps, polar_dtype=polar_dtype, **kwargs)


class AMUSEAuroraFast2(AMUSEAurora):
    """More aggressive fast variant: factored preconditioning with few polar steps.

    At a small step budget the single fixed "simple" quintic (3.4445, -4.7750,
    2.0315) under-orthogonalizes, because those coefficients only converge after
    ~5 steps. This variant instead drives the Newton-Schulz iteration with the
    modded-nanogpt speedrun's per-step schedule, whose composed polynomials lift the
    singular values toward 1 much faster. In a polar-factor-distance benchmark it is
    never worse than the fixed coefficients and clearly better on higher-rank
    matrices, while remaining monotone and stable -- unlike the more aggressive Polar
    Express / CANS schedules, which overshoot and diverge at 2-3 steps. Compute is
    bf16 rather than fp16: the leading coefficients produce large intermediate
    magnitudes that can overflow fp16's narrow range, and bf16 matches the precision
    the speedrun coefficients were tuned in.
    """

    def __init__(self, *args, polar_steps=2, polar_dtype=torch.bfloat16, **kwargs):
        kwargs.setdefault("factored_second_moment", True)
        kwargs.setdefault("track_update_cosine", False)
        kwargs.setdefault("polar_coeffs", MODDED_NANOGPT_NS_COEFFS)
        super().__init__(*args, polar_steps=polar_steps, polar_dtype=polar_dtype, **kwargs)


class ScheduleFreeAdamW:
    """Schedule-free AdamW baseline with explicit z/x state."""

    def __init__(
        self,
        named_params,
        lr=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        sf_beta1=0.6,
        rho=0.8,
        warmup_steps=100,
        fixed_beta=None,
        debug_checks=False,
        track_update_cosine=True,
    ):
        self.param_entries = _as_named_entries(named_params)
        self.params = [param for _, param in self.param_entries]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.sf_beta1 = sf_beta1
        self.rho = rho
        self.warmup_steps = max(1, int(warmup_steps))
        self.fixed_beta = fixed_beta
        self.debug_checks = debug_checks
        self.track_update_cosine = track_update_cosine
        self.step_idx = 0
        self.c_t = 0.0
        self.c_t0 = None
        self.last_beta = self.beta_t()
        self.last_update_cosine = float("nan")
        self.last_raw_cosine = float("nan")
        self.last_reconstruction_error = float("nan")
        self._x_prev = None
        self._x_prev_prev = None

        self.state = {}
        for name, p in self.param_entries:
            self.state[p] = {
                "name": name,
                "z": p.detach().clone(),
                "x": p.detach().clone(),
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }
        if self.track_update_cosine:
            self._x_prev = self._flatten_x_snapshot()

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None if set_to_none else torch.zeros_like(p)

    def beta_t(self):
        if self.fixed_beta is not None:
            return float(self.fixed_beta)
        if self.step_idx <= self.warmup_steps or self.c_t0 is None or self.c_t <= self.c_t0:
            return float(self.sf_beta1)
        # same schedule-free beta rule here: anchor at c_t0 after warmup
        beta = 1.0 - (self.c_t0 / self.c_t) ** self.rho * (1.0 - self.sf_beta1)
        return float(min(max(beta, self.sf_beta1), 1.0))

    @torch.no_grad()
    def prepare_train_step(self):
        beta = self.beta_t()
        self.last_beta = beta
        for _, p in self.param_entries:
            state = self.state[p]
            p.copy_(state["z"])
            p.lerp_(state["x"], beta)

    @torch.no_grad()
    def use_x_params(self):
        for _, p in self.param_entries:
            p.copy_(self.state[p]["x"])

    @torch.no_grad()
    def use_z_params(self):
        for _, p in self.param_entries:
            p.copy_(self.state[p]["z"])

    @torch.no_grad()
    def use_y_params(self):
        self.prepare_train_step()

    @torch.no_grad()
    def step(self):
        lr_sq = self.lr * self.lr
        # schedule-free averaging here also uses lr^2 weights
        next_c_t = self.c_t + lr_sq
        x_weight = 0.0 if next_c_t == 0.0 else lr_sq / next_c_t

        for _, p in self.param_entries:
            if p.grad is None:
                continue
            state = self.state[p]
            z = state["z"]
            grad = p.grad
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            z.mul_(1.0 - self.lr * self.weight_decay)
            exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

            t = self.step_idx + 1
            bias_correction1 = 1.0 - self.beta1**t
            bias_correction2 = 1.0 - self.beta2**t
            step_size = self.lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
            z.addcdiv_(exp_avg, denom, value=-step_size)
            state["x"].lerp_(z, x_weight)

        self.c_t = next_c_t
        if self.step_idx + 1 == self.warmup_steps and self.c_t0 is None:
            # freeze the warmup boundary in c-space, then grow beta from there
            self.c_t0 = self.c_t

        if self.track_update_cosine:
            self.last_update_cosine = self._update_cosine()
        self.step_idx += 1
        self.prepare_train_step()

    def _flatten_x_snapshot(self):
        return torch.cat(
            [self.state[param]["x"].detach().double().flatten().cpu() for _, param in self.param_entries]
        )

    def _update_cosine(self):
        x_now = self._flatten_x_snapshot()
        if self._x_prev is None:
            self._x_prev = x_now
            return float("nan")
        if self._x_prev_prev is None:
            self._x_prev_prev = self._x_prev
            self._x_prev = x_now
            return float("nan")

        dx = x_now - self._x_prev
        dx_prev = self._x_prev - self._x_prev_prev
        denom = dx.norm() * dx_prev.norm()
        if denom.item() < 1e-20:
            raw_cosine = float("nan")
            cosine = float("nan")
        else:
            raw_cosine = torch.dot(dx, dx_prev).div(denom).item()
            if self.debug_checks and (raw_cosine > 1.0001 or raw_cosine < -1.0001):
                print(
                    f"[sf-adamw warning] step={self.step_idx} raw_cosine={raw_cosine:.8f} "
                    f"dx_norm={dx.norm().item():.3e} prev_dx_norm={dx_prev.norm().item():.3e}"
                )
            cosine = float(torch.clamp(torch.tensor(raw_cosine, dtype=torch.float64), -1.0, 1.0).item())

        self.last_raw_cosine = raw_cosine
        self._x_prev_prev = self._x_prev
        self._x_prev = x_now
        return cosine

    @torch.no_grad()
    def metrics(self):
        zx_num = zx_den = yx_num = yx_den = yz_num = yz_den = 0.0
        reconstruction_num = reconstruction_den = 0.0
        beta = self.last_beta
        for _, p in self.param_entries:
            state = self.state[p]
            z = state["z"].to(torch.float64)
            x = state["x"].to(torch.float64)
            y = torch.lerp(z, x, beta)
            model_y = p.detach().double()
            zx_delta = z - x
            yx_delta = y - x
            yz_delta = y - z
            zx_num += torch.sum(zx_delta.pow(2)).item()
            zx_den += torch.sum(x.pow(2)).item()
            yx_num += torch.sum(yx_delta.pow(2)).item()
            yx_den += torch.sum(x.pow(2)).item()
            yz_num += torch.sum(yz_delta.pow(2)).item()
            yz_den += torch.sum(z.pow(2)).item()
            reconstruction_num += torch.sum((model_y - y).pow(2)).item()
            reconstruction_den += torch.sum(model_y.pow(2)).item()
        zx_distance = _relative_norm(zx_num, zx_den, 1e-12)
        yx_distance = _relative_norm(yx_num, yx_den, 1e-12)
        yz_distance = _relative_norm(yz_num, yz_den, 1e-12)
        self.last_reconstruction_error = _relative_norm(reconstruction_num, reconstruction_den, 1e-12)
        if self.debug_checks and self.last_reconstruction_error > 1e-8:
            print(f"[sf-adamw warning] step={self.step_idx} reconstruction_error={self.last_reconstruction_error:.3e}")
        return {
            "beta_t": beta,
            "update_cosine_similarity": self.last_update_cosine,
            "z_x_distance": zx_distance,
            "y_x_distance": yx_distance,
            "y_z_distance": yz_distance,
            "y_x_expected_distance": (1.0 - beta) * zx_distance,
            "y_z_expected_distance": beta * _relative_norm(zx_num, yz_den, 1e-12),
            "amuse_y_reconstruction_error": self.last_reconstruction_error,
        }
