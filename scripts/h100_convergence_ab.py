"""Controlled convergence A/B on real FineWeb GPT-2 tokens.

Only ONE thing differs between the two runs: the Muon orthogonalization.
  - 'wr'   : Polar Express, 5 iterations, exact modded-nanogpt WR coefficients.
  - 'ours' : modded-nanogpt quintic schedule, 2 iterations (our fast2 change).
Everything else (model init seed, data order, LR, momentum, steps) is identical,
so any difference in val loss is attributable to the orthogonalization quality.
"""
import sys, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 256
dev = torch.device("cuda")
DATA = "/root/modded-nanogpt/data/fineweb10B"

# ---------------- data ----------------
def load_shard(path):
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "bad magic"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(ntok * 2), dtype=np.uint16)
    return torch.from_numpy(tokens.astype(np.int64))

def batches(tokens, B, T, seed):
    g = torch.Generator().manual_seed(seed)
    n = (len(tokens) - 1) // (B * T)
    while True:
        for i in torch.randperm(n, generator=g).tolist():
            chunk = tokens[i * B * T : (i + 1) * B * T + 1]
            x = chunk[:-1].view(B, T).to(dev, non_blocking=True)
            y = chunk[1:].view(B, T).to(dev, non_blocking=True)
            yield x, y

# ---------------- orthogonalization ----------------
WR_PE = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
QUINTIC = [
    (4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012),
]
def orthogonalize(G, mode):
    sched = WR_PE[:5] if mode == "wr" else QUINTIC[:2]
    cushion = 2e-2 if mode == "wr" else 0.0
    X = G.bfloat16()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.mT
    X = X / (X.norm() * (1 + cushion) + 1e-6)
    for a, b, c in sched:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return (X.mT if tall else X).to(G.dtype)

class Muon:
    """Single-GPU Muon for 2D params; Adam for the rest. Orthogonalization mode swappable."""
    def __init__(self, matrix_params, other_params, mode, lr=0.02, momentum=0.95, adam_lr=3e-3):
        self.mp = list(matrix_params); self.op = list(other_params)
        self.mode = mode; self.lr = lr; self.momentum = momentum; self.adam_lr = adam_lr
        self.mbuf = [torch.zeros_like(p) for p in self.mp]
        self.a_m = [torch.zeros_like(p) for p in self.op]
        self.a_v = [torch.zeros_like(p) for p in self.op]
        self.t = 0
    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, buf in zip(self.mp, self.mbuf):
            if p.grad is None: continue
            buf.lerp_(p.grad, 1 - self.momentum)
            g = p.grad.lerp(buf, self.momentum)  # nesterov
            u = orthogonalize(g, self.mode)
            # Normalize BOTH modes to the same target Frobenius norm sqrt(min(m,n))
            # (the magnitude a perfectly-orthogonal update would have). This removes
            # the effective-LR confound: a 2-step update is slightly shorter than a
            # 5-step one, so without this, "ours" would just be using a different
            # step size. Now the ONLY difference between modes is the direction.
            target = math.sqrt(min(p.size(0), p.size(1)))
            u = u / u.norm().clamp_min(1e-7) * target
            u = u * max(1.0, p.size(0) / p.size(1)) ** 0.5
            p.add_(u, alpha=-self.lr)
        b1, b2, eps = 0.9, 0.95, 1e-10
        for p, m, v in zip(self.op, self.a_m, self.a_v):
            if p.grad is None: continue
            m.lerp_(p.grad, 1 - b1); v.lerp_(p.grad.square(), 1 - b2)
            mh = m / (1 - b1 ** self.t); vh = v / (1 - b2 ** self.t)
            p.add_(mh / (vh.sqrt() + eps), alpha=-self.adam_lr)
    def zero_grad(self):
        for p in self.mp + self.op:
            p.grad = None

# ---------------- model ----------------
class Block(nn.Module):
    def __init__(self, dim, nh):
        super().__init__()
        self.nh = nh
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.fc = nn.Linear(dim, 4 * dim, bias=False)
        self.fcp = nn.Linear(4 * dim, dim, bias=False)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split(C, dim=2)
        q = q.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, C))
        x = x + self.fcp(F.relu(self.fc(self.n2(x))).square())
        return x

class GPT(nn.Module):
    def __init__(self, vocab=50304, dim=512, nl=6, nh=8, T=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, nh) for _ in range(nl)])
        self.nf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight.data.zero_()
        self.pos = nn.Parameter(torch.zeros(1, T, dim))
    def forward(self, idx, tgt):
        x = self.embed(idx) + self.pos[:, :idx.size(1)]
        for b in self.blocks: x = b(x)
        logits = self.head(self.nf(x))
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt.view(-1))

def split_params(model):
    mat, oth = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in n and "head" not in n and "pos" not in n:
            mat.append(p)
        else:
            oth.append(p)
    return mat, oth

@torch.no_grad()
def evaluate(model, val_tokens, B, T, nb=20):
    model.eval()
    g = batches(val_tokens, B, T, seed=1234)
    tot = 0.0
    for _ in range(nb):
        x, y = next(g)
        tot += model(x, y).item()
    model.train()
    return tot / nb

def run(mode, steps, train_tokens, val_tokens, B, T, seed):
    torch.manual_seed(seed)
    model = GPT().to(dev).bfloat16()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm): m.float()
    model = torch.compile(model)
    mat, oth = split_params(model)
    opt = Muon(mat, oth, mode=mode)
    data = batches(train_tokens, B, T, seed=777)
    # warmup/compile
    x, y = next(data); loss = model(x, y); loss.backward(); opt.zero_grad()
    torch.cuda.synchronize()
    log = []
    t0 = time.perf_counter(); step_t0 = t0
    for s in range(1, steps + 1):
        x, y = next(data)
        loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad()
        if s % 50 == 0 or s == steps:
            torch.cuda.synchronize()
            vl = evaluate(model, val_tokens, B, T)
            dt = (time.perf_counter() - step_t0) / (50 if s % 50 == 0 else (s % 50 or 50))
            log.append((s, loss.item(), vl, dt * 1000))
            print(f"  [{mode}] step {s:4d}  train {loss.item():.4f}  val {vl:.4f}  {dt*1000:.1f} ms/step", flush=True)
            step_t0 = time.perf_counter()
    total = time.perf_counter() - t0
    return log, total

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    B, T, seed = 24, 1024, 1234
    print(f"loading data... steps={steps} B={B} T={T}", flush=True)
    train = load_shard(f"{DATA}/fineweb_train_000001.bin")
    val = load_shard(f"{DATA}/fineweb_val_000000.bin")
    print(f"train tokens {len(train)/1e6:.0f}M, val {len(val)/1e6:.0f}M", flush=True)
    results = {}
    for mode in ["wr", "ours"]:
        print(f"\n=== run mode={mode} ===", flush=True)
        log, total = run(mode, steps, train, val, B, T, seed)
        results[mode] = (log, total)
    print("\n================ SUMMARY ================")
    for mode in ["wr", "ours"]:
        log, total = results[mode]
        s, tr, vl, ms = log[-1]
        avg_ms = sum(r[3] for r in log) / len(log)
        print(f"{mode:>5}: final val {vl:.4f} | avg {avg_ms:.1f} ms/step | wall {total:.1f}s")
    (lw, tw), (lo, to) = results["wr"], results["ours"]
    print(f"\nval loss delta (ours - wr): {lo[-1][2]-lw[-1][2]:+.4f}")
    print(f"per-step speedup (wr_ms/ours_ms): {(sum(r[3] for r in lw)/len(lw))/(sum(r[3] for r in lo)/len(lo)):.3f}x")
    print("OK_DONE")
