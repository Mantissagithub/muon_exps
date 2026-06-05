import argparse
import time

import torch


@torch.compile(dynamic=False, fullgraph=True)
def normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2: float, red_dim: int):
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


@torch.compile(dynamic=False, fullgraph=True)
def aurora_row_balance(v_chunk):
    row_sq = v_chunk.float().square().mean(dim=-1, keepdim=True).clamp_min_(1e-10)
    target = row_sq.mean(dim=-2, keepdim=True)
    scale = (target / row_sq).sqrt_().type_as(v_chunk)
    return v_chunk.mul_(scale)


def row_cv(x):
    row_norm = x.float().square().mean(dim=-1).sqrt()
    return (row_norm.std(dim=-1) / row_norm.mean(dim=-1).clamp_min(1e-10)).mean().item()


def bench(fn, iters):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rows", type=int, default=768)
    p.add_argument("--cols", type=int, default=768)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--beta2", type=float, default=0.95)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this speedrun probe")

    device = torch.device("cuda")
    shape = (args.batch, args.rows, args.cols)
    red_dim = -1 if args.rows >= args.cols else -2
    second_shape = (args.batch, args.rows, 1) if red_dim == -1 else (args.batch, 1, args.cols)

    torch.manual_seed(1337)
    base = torch.randn(shape, device=device, dtype=torch.bfloat16)
    second = torch.zeros(second_shape, device=device, dtype=torch.float32)

    for _ in range(args.warmup):
        x = base.clone()
        normuon_variance_reduction(x, second, args.beta2, red_dim)
    torch.cuda.synchronize()

    baseline_ms = bench(
        lambda: normuon_variance_reduction(base.clone(), second, args.beta2, red_dim),
        args.iters,
    )

    for _ in range(args.warmup):
        x = base.clone()
        normuon_variance_reduction(x, second, args.beta2, red_dim)
        aurora_row_balance(x)
    torch.cuda.synchronize()

    aurora_ms = bench(
        lambda: aurora_row_balance(normuon_variance_reduction(base.clone(), second, args.beta2, red_dim)),
        args.iters,
    )

    x0 = normuon_variance_reduction(base.clone(), second, args.beta2, red_dim)
    x1 = aurora_row_balance(x0.clone())
    print(f"shape={shape} red_dim={red_dim}")
    print(f"baseline_normuon_ms={baseline_ms:.4f}")
    print(f"with_aurora_balance_ms={aurora_ms:.4f}")
    print(f"extra_ms={aurora_ms - baseline_ms:.4f}")
    print(f"extra_pct={(aurora_ms / baseline_ms - 1.0) * 100:.2f}")
    print(f"row_cv_before={row_cv(x0):.6f}")
    print(f"row_cv_after={row_cv(x1):.6f}")


if __name__ == "__main__":
    main()
