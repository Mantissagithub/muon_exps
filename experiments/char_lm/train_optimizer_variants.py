import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.char_lm.dataset import CharDataset, load_tinyshakespeare
from experiments.char_lm.model import TinyCharTransformer
from optimizers import AMUSEMuon, Aurora, MuonLike, NorMuon, RiemannAurora, UNorMuon


OPTIMIZERS = [
    "adamw",
    "torch_muon",
    "muon_like",
    "normuon",
    "u_normuon",
    "aurora",
    "riemann_aurora",
    "amuse_muon",
    "sf_muon_fixed_beta_0.6",
    "sf_muon_fixed_beta_0.9",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--amuse-beta1", type=float, default=0.6)
    p.add_argument("--amuse-rho", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "tinyshakespeare" / "input.txt")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "char_lm")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_params(model: torch.nn.Module):
    matrix_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_hidden_matrix = param.ndim == 2 and name.endswith("weight")
        is_excluded = name.startswith("token_embedding_table") or name.startswith("position_embedding_table") or name.startswith("lm_head")
        if is_hidden_matrix and not is_excluded:
            matrix_params.append(param)
        else:
            other_params.append(param)
    return matrix_params, other_params


def make_optimizers(name: str, model: torch.nn.Module, lr: float, weight_decay: float):
    matrix_params, other_params = split_params(model)
    if name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)]

    other_opt = torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay) if other_params else None
    if name == "torch_muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("this torch build does not expose torch.optim.Muon")
        matrix_opt = torch.optim.Muon(matrix_params, lr=lr, weight_decay=weight_decay)
    elif name == "muon_like":
        matrix_opt = MuonLike(matrix_params, lr=lr, weight_decay=weight_decay)
    elif name == "normuon":
        matrix_opt = NorMuon(matrix_params, lr=lr, weight_decay=weight_decay)
    elif name == "u_normuon":
        matrix_opt = UNorMuon(matrix_params, lr=lr, weight_decay=weight_decay)
    elif name == "aurora":
        matrix_opt = Aurora(matrix_params, lr=lr, weight_decay=weight_decay)
    elif name == "riemann_aurora":
        matrix_opt = RiemannAurora(matrix_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {name}")
    return [opt for opt in [matrix_opt, other_opt] if opt is not None]


def make_optimizer_stack(name: str, model: torch.nn.Module, args):
    matrix_params, other_params = split_params(model)
    if name == "amuse_muon":
        return [
            AMUSEMuon(
                matrix_params,
                other_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                beta1=args.amuse_beta1,
                rho=args.amuse_rho,
                warmup_steps=args.warmup_steps,
            )
        ]
    if name == "sf_muon_fixed_beta_0.6":
        return [
            AMUSEMuon(
                matrix_params,
                other_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                beta1=0.6,
                rho=args.amuse_rho,
                warmup_steps=args.warmup_steps,
                fixed_beta=0.6,
            )
        ]
    if name == "sf_muon_fixed_beta_0.9":
        return [
            AMUSEMuon(
                matrix_params,
                other_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                beta1=0.9,
                rho=args.amuse_rho,
                warmup_steps=args.warmup_steps,
                fixed_beta=0.9,
            )
        ]
    return make_optimizers(name, model, args.lr, args.weight_decay)


def find_amuse(optimizers):
    for opt in optimizers:
        if isinstance(opt, AMUSEMuon):
            return opt
    return None


@torch.no_grad()
def estimate_loss(model, data, batch_size, device, eval_iters, seed):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        gen = torch.Generator().manual_seed(seed + (0 if split == "train" else 10_000))
        losses = torch.empty(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(split, batch_size, device, gen)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def estimate_val_loss(model, data, batch_size, device, eval_iters, seed):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = torch.empty(eval_iters)
    for k in range(eval_iters):
        x, y = data.get_batch("val", batch_size, device, gen)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def run_one(name, base_state, data, args, device):
    set_seed(args.seed)
    model = TinyCharTransformer(
        vocab_size=data.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(base_state)
    optimizers = make_optimizer_stack(name, model, args)
    amuse = find_amuse(optimizers)
    if amuse is not None:
        amuse.prepare_train_step()
    train_gen = torch.Generator().manual_seed(args.seed)
    rows = []
    best_val = float("inf")
    t0 = time.perf_counter()
    tokens_seen = 0
    last_step_time = float("nan")

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            val_loss_z = float("nan")
            if amuse is not None:
                amuse.use_x_params()
            losses = estimate_loss(model, data, args.batch_size, device, args.eval_iters, args.seed + step)
            if amuse is not None:
                amuse.use_z_params()
                val_loss_z = estimate_val_loss(model, data, args.batch_size, device, args.eval_iters, args.seed + step + 10_000)
                amuse.prepare_train_step()
            best_val = min(best_val, losses["val"])
            elapsed = time.perf_counter() - t0
            metrics = amuse.metrics() if amuse is not None else {}
            rows.append({
                "optimizer": name,
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "val_loss_z": val_loss_z,
                "beta_t": metrics.get("beta_t", float("nan")),
                "lr": args.lr,
                "step_time": last_step_time,
                "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
                "elapsed_s": elapsed,
                "best_val_loss": best_val,
                "update_cosine_similarity": metrics.get("update_cosine_similarity", float("nan")),
                "z_x_distance": metrics.get("z_x_distance", float("nan")),
                "y_x_distance": metrics.get("y_x_distance", float("nan")),
                "y_z_distance": metrics.get("y_z_distance", float("nan")),
            })
            print(f"{name},{step},train={losses['train']:.4f},val={losses['val']:.4f},best={best_val:.4f}")
        if step == args.steps:
            break

        step_start = time.perf_counter()
        x, y = data.get_batch("train", args.batch_size, device, train_gen)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        for opt in optimizers:
            opt.step()
        tokens_seen += args.batch_size * args.block_size
        last_step_time = time.perf_counter() - step_start

    return rows


def write_results(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    fields = [
        "optimizer",
        "step",
        "train_loss",
        "val_loss",
        "val_loss_z",
        "beta_t",
        "lr",
        "step_time",
        "tokens_per_sec",
        "elapsed_s",
        "best_val_loss",
        "update_cosine_similarity",
        "z_x_distance",
        "y_x_distance",
        "y_z_distance",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_opt = {}
    for row in rows:
        by_opt.setdefault(row["optimizer"], []).append(row)
    lines = ["# char lm optimizer results", ""]
    for opt, opt_rows in by_opt.items():
        last = opt_rows[-1]
        lines.append(f"- `{opt}`: best val {last['best_val_loss']:.4f}, final val {last['val_loss']:.4f}, {last['elapsed_s']:.1f}s")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def main():
    args = parse_args()
    requested = [x.strip() for x in args.optimizers.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(OPTIMIZERS))
    if unknown:
        raise ValueError(f"unknown optimizers: {unknown}")
    if "torch_muon" in requested and not hasattr(torch.optim, "Muon"):
        print("skipping torch_muon: this torch build does not expose torch.optim.Muon")
        requested = [name for name in requested if name != "torch_muon"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = load_tinyshakespeare(args.dataset)
    data = CharDataset(text, block_size=args.block_size)

    set_seed(args.seed)
    base_model = TinyCharTransformer(
        vocab_size=data.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    all_rows = []
    for name in requested:
        all_rows.extend(run_one(name, base_state, data, args, device))

    csv_path = write_results(all_rows, args.out_dir)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
