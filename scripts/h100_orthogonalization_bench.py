"""H100 orthogonalization benchmark, faithful to the current modded-nanogpt WR.

WR baseline = Polar Express, 5 iterations, exact WR coeffs (safety_factor=2e-2),
spectral-bound (Frobenius*(1+2e-2)) normalization. We compare:
  - WR PE-5          (what the record runs)
  - ours quintic-2   (our amuse_aurora_fast2 change: 2-step modded-nanogpt quintic)
  - PE-2, PE-3       (Polar Express truncated, for reference)
Reports per-call ms (bf16, torch.compiled) and relative distance to the true
polar factor U V^T (lower = better orthogonalization).
"""
import torch, time
torch._dynamo.config.cache_size_limit = 256

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WR_PE = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
QUINTIC = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]

def make_ns(coeffs, steps, cushion):
    sched = coeffs[:steps] if len(coeffs) >= steps else coeffs + [coeffs[-1]] * (steps - len(coeffs))
    @torch.compile(dynamic=False, fullgraph=True)
    def ns(G):
        X = G.bfloat16()
        tall = X.size(-2) > X.size(-1)
        if tall:
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + cushion) + 1e-6)
        for a, b, c in sched:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return X.mT if tall else X
    return ns

def polar_dist(P, A):
    U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    t = U @ Vh
    return (P.float() - t).norm().item() / t.norm().item()

def bench(fn, G, iters=100, warmup=25):
    for _ in range(warmup):
        fn(G)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(G)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

variants = {
    "WR_PE5":    make_ns(WR_PE, 5, 2e-2),
    "ours_q2":   make_ns(QUINTIC, 2, 0.0),
    "PE2":       make_ns(WR_PE, 2, 2e-2),
    "PE3":       make_ns(WR_PE, 3, 2e-2),
}
print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
shapes = [(768,768),(768,3072),(3072,768),(1024,1024),(1024,4096),(4096,1024)]
torch.manual_seed(0)
ms = {k: 0.0 for k in variants}
print(f"{'shape':>12} " + " ".join(f"{k+'_ms':>11}" for k in variants))
for (m,n) in shapes:
    G = torch.randn(m, n, device=dev)
    row = []
    for k, fn in variants.items():
        t = bench(fn, G); ms[k] += t; row.append(f"{t:>11.4f}")
    print(f"{str((m,n)):>12} " + " ".join(row))
print("\nTotal ms over shapes & speedup vs WR_PE5:")
for k in variants:
    print(f"  {k:>8}: {ms[k]:>8.4f} ms   {ms['WR_PE5']/ms[k]:>5.2f}x")
print(f"\n{'shape':>12} " + " ".join(f"{k+'_err':>11}" for k in variants))
for (m,n) in shapes:
    G = torch.randn(m, n, device=dev)
    row = [f"{polar_dist(fn(G), G):>11.3e}" for fn in variants.values()]
    print(f"{str((m,n)):>12} " + " ".join(row))
print("OK_DONE")
