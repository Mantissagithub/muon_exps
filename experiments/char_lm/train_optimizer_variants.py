import argparse
import csv
import importlib
import math
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
from optimizers import AMUSEMuon, Aurora, MuonLike, NorMuon, RiemannAurora, ScheduleFreeAdamW, UNorMuon


AMUSE_VARIANTS = {
    "amuse_muon_b0.4_r0.5": {"beta1": 0.4, "rho": 0.5, "fixed_beta": float("nan")},
    "amuse_muon_b0.4_r0.8": {"beta1": 0.4, "rho": 0.8, "fixed_beta": float("nan")},
    "amuse_muon_b0.6_r0.5": {"beta1": 0.6, "rho": 0.5, "fixed_beta": float("nan")},
    "amuse_muon_b0.6_r0.8": {"beta1": 0.6, "rho": 0.8, "fixed_beta": float("nan")},
    "sf_muon_fixed_beta_0.6": {"beta1": 0.6, "rho": float("nan"), "fixed_beta": 0.6},
    "sf_muon_fixed_beta_0.9": {"beta1": 0.9, "rho": float("nan"), "fixed_beta": 0.9},
}


OPTIMIZERS = [
    "adamw",
    "torch_muon",
    "sf_adamw",
    "amuse_muon",
    *AMUSE_VARIANTS.keys(),
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
    p.add_argument("--amuse-lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--amuse-beta1", type=float, default=0.4)
    p.add_argument("--amuse-rho", type=float, default=0.5)
    p.add_argument("--amuse-debug-checks", action="store_true")
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
    matrix_named_params, other_named_params = split_named_params(model)
    return [param for _, param in matrix_named_params], [param for _, param in other_named_params]


def split_named_params(model: torch.nn.Module):
    matrix_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_hidden_matrix = param.ndim == 2 and name.endswith("weight")
        is_excluded = name.startswith("token_embedding_table") or name.startswith("position_embedding_table") or name.startswith("lm_head")
        if is_hidden_matrix and not is_excluded:
            matrix_params.append((name, param))
        else:
            other_params.append((name, param))
    return matrix_params, other_params


def make_optimizers(name: str, model: torch.nn.Module, lr: float, weight_decay: float):
    matrix_named_params, other_named_params = split_named_params(model)
    matrix_params = [param for _, param in matrix_named_params]
    other_params = [param for _, param in other_named_params]
    if name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)]
    if name == "adafactor":
        if not hasattr(torch.optim, "Adafactor"):
            raise RuntimeError("this torch build does not expose torch.optim.Adafactor")
        return [torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay)]

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
    elif name in {"lion", "adopt", "prodigy", "soap", "mars", "sophia", "ademamix"}:
        return [make_optional_optimizer(name, model, lr, weight_decay)]
    else:
        raise ValueError(f"unknown optimizer: {name}")
    return [opt for opt in [matrix_opt, other_opt] if opt is not None]


def make_optional_optimizer(name: str, model: torch.nn.Module, lr: float, weight_decay: float):
    specs = {
        "lion": ("lion_pytorch", "Lion"),
        "adopt": ("adopt", "ADOPT"),
        "prodigy": ("prodigyopt", "Prodigy"),
        "soap": ("soap", "SOAP"),
        "mars": ("mars", "MARS"),
        "sophia": ("sophia", "Sophia"),
        "ademamix": ("ademamix", "AdEMAMix"),
    }
    module_name, class_name = specs[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{name} requested but module '{module_name}' is not installed in .venv") from exc
    opt_cls = getattr(module, class_name)
    return opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_optimizer_stack(name: str, model: torch.nn.Module, args):
    matrix_named_params, other_named_params = split_named_params(model)
    variant = AMUSE_VARIANTS.get(name)
    if name == "sf_adamw":
        return [
            ScheduleFreeAdamW(
                list(model.named_parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
                sf_beta1=args.amuse_beta1,
                rho=args.amuse_rho,
                warmup_steps=args.warmup_steps,
                debug_checks=args.amuse_debug_checks,
            )
        ]
    if name == "amuse_muon":
        return [
            AMUSEMuon(
                matrix_named_params,
                other_named_params,
                lr=args.amuse_lr,
                weight_decay=args.weight_decay,
                beta1=args.amuse_beta1,
                rho=args.amuse_rho,
                warmup_steps=args.warmup_steps,
                debug_checks=args.amuse_debug_checks,
            )
        ]
    if variant is not None:
        return [
            AMUSEMuon(
                matrix_named_params,
                other_named_params,
                lr=args.amuse_lr,
                weight_decay=args.weight_decay,
                beta1=variant["beta1"],
                rho=variant["rho"] if math.isfinite(variant["rho"]) else args.amuse_rho,
                warmup_steps=args.warmup_steps,
                fixed_beta=variant["fixed_beta"] if math.isfinite(variant["fixed_beta"]) else None,
                debug_checks=args.amuse_debug_checks,
            )
        ]
    return make_optimizers(name, model, args.lr, args.weight_decay)


def find_amuse(optimizers):
    for opt in optimizers:
        if isinstance(opt, (AMUSEMuon, ScheduleFreeAdamW)):
            return opt
    return None


def optimizer_metadata(name: str, args):
    variant = AMUSE_VARIANTS.get(name)
    if name == "amuse_muon":
        return {"beta1": args.amuse_beta1, "rho": args.amuse_rho, "fixed_beta": float("nan")}
    if name == "sf_adamw":
        return {"beta1": args.amuse_beta1, "rho": args.amuse_rho, "fixed_beta": float("nan")}
    if variant is not None:
        return variant
    return {"beta1": float("nan"), "rho": float("nan"), "fixed_beta": float("nan")}


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
    meta = optimizer_metadata(name, args)
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
            val_loss_y = float("nan")
            val_loss_x = float("nan")
            if amuse is not None:
                # main reported validation is always on x_t
                amuse.use_x_params()
            losses = estimate_loss(model, data, args.batch_size, device, args.eval_iters, args.seed + step)
            val_loss_x = losses["val"]
            if amuse is not None:
                amuse.use_z_params()
                val_loss_z = estimate_val_loss(model, data, args.batch_size, device, args.eval_iters, args.seed + step + 10_000)
                amuse.use_y_params()
                val_loss_y = estimate_val_loss(model, data, args.batch_size, device, args.eval_iters, args.seed + step + 20_000)
            best_val = min(best_val, val_loss_x)
            elapsed = time.perf_counter() - t0
            metrics = amuse.metrics() if amuse is not None else {}
            rows.append({
                "optimizer": name,
                "beta1": meta["beta1"],
                "rho": meta["rho"],
                "fixed_beta": meta["fixed_beta"],
                "step": step,
                "train_loss": losses["train"],
                "val_loss": val_loss_x,
                "val_loss_x": val_loss_x,
                "val_loss_z": val_loss_z,
                "val_loss_y": val_loss_y,
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
                "y_x_expected_distance": metrics.get("y_x_expected_distance", float("nan")),
                "y_z_expected_distance": metrics.get("y_z_expected_distance", float("nan")),
                "amuse_y_reconstruction_error": metrics.get("amuse_y_reconstruction_error", float("nan")),
            })
            if amuse is not None:
                amuse.use_y_params()
            print(f"{name},{step},train={losses['train']:.4f},val_x={val_loss_x:.4f},best={best_val:.4f}")
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
        "beta1",
        "rho",
        "fixed_beta",
        "step",
        "train_loss",
        "val_loss",
        "val_loss_x",
        "val_loss_z",
        "val_loss_y",
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
        "y_x_expected_distance",
        "y_z_expected_distance",
        "amuse_y_reconstruction_error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_opt = {}
    for row in rows:
        by_opt.setdefault(row["optimizer"], []).append(row)
    lines = [
        "# char lm optimizer results",
        "",
        "| optimizer | beta1 | rho | fixed_beta | best_val_loss | train_at_best | final_val_loss | final_train_loss | step_of_best_val | wall_time | avg_update_cosine | final_zx_distance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for opt, opt_rows in sorted(by_opt.items()):
        last = opt_rows[-1]
        best_row = min(opt_rows, key=lambda row: row["val_loss"])
        finite_cosines = [row["update_cosine_similarity"] for row in opt_rows if math.isfinite(row["update_cosine_similarity"])]
        avg_cosine = sum(finite_cosines) / len(finite_cosines) if finite_cosines else float("nan")
        lines.append(
            "| "
            f"`{opt}` | "
            f"{last['beta1'] if math.isfinite(last['beta1']) else ''} | "
            f"{last['rho'] if math.isfinite(last['rho']) else ''} | "
            f"{last['fixed_beta'] if math.isfinite(last['fixed_beta']) else ''} | "
            f"{best_row['best_val_loss']:.4f} | "
            f"{best_row['train_loss']:.4f} | "
            f"{last['val_loss']:.4f} | "
            f"{last['train_loss']:.4f} | "
            f"{best_row['step']} | "
            f"{last['elapsed_s']:.1f}s | "
            f"{avg_cosine if math.isfinite(avg_cosine) else float('nan'):.6f} | "
            f"{last['z_x_distance'] if math.isfinite(last['z_x_distance']) else float('nan'):.6f} |"
        )
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
    if "adafactor" in requested and not hasattr(torch.optim, "Adafactor"):
        print("skipping adafactor: this torch build does not expose torch.optim.Adafactor")
        requested = [name for name in requested if name != "adafactor"]
    optional_modules = {
        "lion": "lion_pytorch",
        "adopt": "adopt",
        "prodigy": "prodigyopt",
        "soap": "soap",
        "mars": "mars",
        "sophia": "sophia",
        "ademamix": "ademamix",
    }
    for name, module_name in optional_modules.items():
        if name in requested:
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                print(f"skipping {name}: module '{module_name}' is not installed in .venv")
                requested = [opt_name for opt_name in requested if opt_name != name]

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
