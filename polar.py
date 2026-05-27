import torch


@torch.no_grad()
def polar(x: torch.Tensor, steps: int = 5, eps: float = 1e-6) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"polar expects a 2D tensor, got shape {tuple(x.shape)}")

    transposed = x.size(0) < x.size(1)
    y = x.mT if transposed else x
    y = y.to(torch.float32)
    y = y / (y.norm() + eps)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        yy_t = y @ y.mT
        y = a * y + b * (yy_t @ y) + c * ((yy_t @ yy_t) @ y)

    y = y.mT if transposed else y
    return y.to(dtype=x.dtype)
