import torch

# Per-step Newton-Schulz schedule from the modded-nanogpt speedrun (Keller Jordan
# et al.). Instead of repeating one fixed quintic p(x) = a*x + b*x^3 + c*x^5, each
# iteration uses its own (a, b, c), tuned together so the *composition* of the five
# polynomials maps a Frobenius-normalized matrix's singular values onto ~1 as fast as
# possible. The early steps have a steep slope at 0 (large a) to lift the small
# singular values that Frobenius normalization produces; the later steps flatten out
# near 1 to suppress overshoot. These are the constants the current speedrun world
# record runs with, and they strictly dominate the single fixed "simple" triple at
# every step count >= 2 in a polar-factor-distance benchmark while staying monotone
# and stable in bf16 (unlike the aggressive Polar Express / CANS schedules, which
# overshoot and diverge at 2-3 steps).
MODDED_NANOGPT_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


@torch.no_grad()
def polar(
    x: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-6,
    compute_dtype: torch.dtype | None = None,
    coeffs: tuple[tuple[float, float, float], ...] | None = None,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"polar expects a 2D tensor, got shape {tuple(x.shape)}")

    transposed = x.size(0) < x.size(1)
    y = x.mT if transposed else x
    dtype = compute_dtype if compute_dtype is not None and x.is_cuda else torch.float32
    y = y.to(dtype)
    y = y / (y.norm() + eps)

    if coeffs is None:
        # Legacy fixed Keller-Jordan quintic (tuned for ~5 steps).
        for _ in range(steps):
            yy_t = y @ y.mT
            y = 3.4445 * y - 4.7750 * (yy_t @ y) + 2.0315 * ((yy_t @ yy_t) @ y)
    else:
        # Per-step optimal schedule; hold the limiting coefficients past its length.
        last = len(coeffs) - 1
        for i in range(steps):
            a, b, c = coeffs[i] if i <= last else coeffs[last]
            yy_t = y @ y.mT
            y = a * y + b * (yy_t @ y) + c * ((yy_t @ yy_t) @ y)

    y = y.mT if transposed else y
    return y.to(dtype=x.dtype)
