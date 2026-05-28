import argparse
import csv
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.char_lm.dataset import CharDataset, load_tinyshakespeare
from experiments.char_lm.model import TinyCharTransformer
from experiments.char_lm.train_optimizer_variants import estimate_loss, split_named_params, write_results


FIELDS = [
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


@dataclass(frozen=True)
class Candidate:
    optimizer: str
    lr: float
    weight_decay: float
    warmup_steps: int
    min_lr_ratio: float
    other_lr: float | None = None

    @property
    def label(self) -> str:
        parts = [
            self.optimizer,
            f"lr={self.lr:g}",
            f"wd={self.weight_decay:g}",
            f"warmup={self.warmup_steps}",
            f"min={self.min_lr_ratio:g}",
        ]
        if self.other_lr is not None:
            parts.insert(2, f"other_lr={self.other_lr:g}")
        return ",".join(parts)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "artifacts" / "char_lm_amuse_fix" / "results.csv")
    p.add_argument("--work-dir", type=Path, default=ROOT / "artifacts" / "char_lm_baseline_tune")
    p.add_argument("--optimizers", default="adamw,torch_muon")
    p.add_argument("--tune-steps", type=int, default=3000)
    p.add_argument("--final-steps", type=int, default=15000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "tinyshakespeare" / "input.txt")
    p.add_argument("--adamw-lrs", default="0.001,0.003,0.005")
    p.add_argument("--muon-lrs", default="0.001,0.003,0.005")
    p.add_argument("--muon-other-lrs", default="0.001,0.003")
    p.add_argument("--weight-decays", default="0.01,0.1")
    p.add_argument("--warmup-steps", default="500,1500")
    p.add_argument("--min-lr-ratios", default="0.1")
    p.add_argument("--skip-tune", action="store_true")
    p.add_argument("--adamw-config")
    p.add_argument("--muon-config")
    return p.parse_args()


def parse_float_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_candidate(optimizer: str, text: str):
    values = {}
    for part in text.split(","):
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return Candidate(
        optimizer=optimizer,
        lr=float(values["lr"]),
        other_lr=float(values["other_lr"]) if "other_lr" in values else None,
        weight_decay=float(values["wd"]),
        warmup_steps=int(values["warmup"]),
        min_lr_ratio=float(values.get("min", 0.1)),
    )


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_base_state(args, data, device):
    set_seed(args.seed)
    base_model = TinyCharTransformer(
        vocab_size=data.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    return {k: v.detach().clone() for k, v in base_model.state_dict().items()}


def make_model(args, data, base_state, device):
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
    return model


def make_optimizers(candidate: Candidate, model):
    if candidate.optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=candidate.lr, weight_decay=candidate.weight_decay)]
    if candidate.optimizer == "torch_muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("this torch build does not expose torch.optim.Muon")
        matrix_named_params, other_named_params = split_named_params(model)
        matrix_params = [param for _, param in matrix_named_params]
        other_params = [param for _, param in other_named_params]
        opts = [torch.optim.Muon(matrix_params, lr=candidate.lr, weight_decay=candidate.weight_decay)]
        if other_params:
            other_lr = candidate.other_lr if candidate.other_lr is not None else candidate.lr
            opts.append(torch.optim.AdamW(other_params, lr=other_lr, weight_decay=candidate.weight_decay))
        return opts
    raise ValueError(f"unsupported optimizer: {candidate.optimizer}")


def make_schedulers(optimizers, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))

    def lr_lambda(step):
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda) for opt in optimizers]


def current_main_lr(optimizers):
    return optimizers[0].param_groups[0]["lr"]


def run_candidate(candidate: Candidate, base_state, data, args, device, steps: int):
    model = make_model(args, data, base_state, device)
    optimizers = make_optimizers(candidate, model)
    schedulers = make_schedulers(optimizers, steps, candidate.warmup_steps, candidate.min_lr_ratio)
    train_gen = torch.Generator().manual_seed(args.seed)
    rows = []
    best_val = float("inf")
    t0 = time.perf_counter()
    tokens_seen = 0
    last_step_time = float("nan")

    for step in range(steps + 1):
        if step % args.eval_interval == 0 or step == steps:
            losses = estimate_loss(model, data, args.batch_size, device, args.eval_iters, args.seed + step)
            best_val = min(best_val, losses["val"])
            elapsed = time.perf_counter() - t0
            rows.append({
                "optimizer": candidate.optimizer,
                "beta1": float("nan"),
                "rho": float("nan"),
                "fixed_beta": float("nan"),
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "val_loss_x": losses["val"],
                "val_loss_z": float("nan"),
                "val_loss_y": float("nan"),
                "beta_t": float("nan"),
                "lr": current_main_lr(optimizers),
                "step_time": last_step_time,
                "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
                "elapsed_s": elapsed,
                "best_val_loss": best_val,
                "update_cosine_similarity": float("nan"),
                "z_x_distance": float("nan"),
                "y_x_distance": float("nan"),
                "y_z_distance": float("nan"),
                "y_x_expected_distance": float("nan"),
                "y_z_expected_distance": float("nan"),
                "amuse_y_reconstruction_error": float("nan"),
            })
            print(
                f"{candidate.label},step={step},"
                f"train={losses['train']:.4f},val={losses['val']:.4f},best={best_val:.4f}"
            )
        if step == steps:
            break

        step_start = time.perf_counter()
        x, y = data.get_batch("train", args.batch_size, device, train_gen)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{candidate.label} produced non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        for opt in optimizers:
            opt.step()
        for scheduler in schedulers:
            scheduler.step()
        tokens_seen += args.batch_size * args.block_size
        last_step_time = time.perf_counter() - step_start

    return rows


def candidate_grid(args, optimizer: str):
    weight_decays = parse_float_list(args.weight_decays)
    warmups = parse_int_list(args.warmup_steps)
    min_ratios = parse_float_list(args.min_lr_ratios)
    if optimizer == "adamw":
        for lr in parse_float_list(args.adamw_lrs):
            for wd in weight_decays:
                for warmup in warmups:
                    for min_ratio in min_ratios:
                        yield Candidate(optimizer, lr, wd, warmup, min_ratio)
    elif optimizer == "torch_muon":
        for lr in parse_float_list(args.muon_lrs):
            for other_lr in parse_float_list(args.muon_other_lrs):
                for wd in weight_decays:
                    for warmup in warmups:
                        for min_ratio in min_ratios:
                            yield Candidate(optimizer, lr, wd, warmup, min_ratio, other_lr=other_lr)
    else:
        raise ValueError(f"unsupported optimizer: {optimizer}")


def summarize_rows(rows):
    best = min(rows, key=lambda row: row["val_loss"])
    final = rows[-1]
    return {
        "best_val_loss": best["val_loss"],
        "best_train_loss": best["train_loss"],
        "best_step": best["step"],
        "final_val_loss": final["val_loss"],
        "final_train_loss": final["train_loss"],
        "final_step": final["step"],
        "elapsed_s": final["elapsed_s"],
    }


def write_tune_summary(path: Path, summaries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fields = [
            "optimizer",
            "candidate",
            "best_val_loss",
            "best_train_loss",
            "best_step",
            "final_val_loss",
            "final_train_loss",
            "final_step",
            "elapsed_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate, summary in summaries:
            writer.writerow({"optimizer": candidate.optimizer, "candidate": candidate.label, **summary})


def load_existing_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def coerce_row(row):
    coerced = {}
    for field in FIELDS:
        value = row.get(field, "")
        if field in {"optimizer"}:
            coerced[field] = value
        elif field == "step":
            coerced[field] = int(value)
        else:
            coerced[field] = float(value)
    return coerced


def splice_results(results_path: Path, replacement_rows):
    targets = {row["optimizer"] for row in replacement_rows}
    existing_rows = [coerce_row(row) for row in load_existing_rows(results_path)]
    kept_rows = [row for row in existing_rows if row["optimizer"] not in targets]
    out_dir = results_path.parent
    write_results(kept_rows + replacement_rows, out_dir)


def tune_optimizer(optimizer, base_state, data, args, device):
    config_arg = args.adamw_config if optimizer == "adamw" else args.muon_config
    if args.skip_tune and config_arg:
        return parse_candidate(optimizer, config_arg)

    summaries = []
    best_candidate = None
    best_summary = None
    for candidate in candidate_grid(args, optimizer):
        rows = run_candidate(candidate, base_state, data, args, device, args.tune_steps)
        summary = summarize_rows(rows)
        summaries.append((candidate, summary))
        if best_summary is None or summary["best_val_loss"] < best_summary["best_val_loss"]:
            best_candidate = candidate
            best_summary = summary
        write_results(rows, args.work_dir / "screens" / candidate.optimizer / candidate.label.replace(",", "__"))

    write_tune_summary(args.work_dir / f"{optimizer}_screen.csv", summaries)
    print(f"selected {best_candidate.label} from {len(summaries)} candidates")
    return best_candidate


def main():
    args = parse_args()
    requested = [name.strip() for name in args.optimizers.split(",") if name.strip()]
    unsupported = sorted(set(requested) - {"adamw", "torch_muon"})
    if unsupported:
        raise ValueError(f"unsupported optimizers: {unsupported}")
    if "torch_muon" in requested and not hasattr(torch.optim, "Muon"):
        raise RuntimeError("this torch build does not expose torch.optim.Muon")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = load_tinyshakespeare(args.dataset)
    data = CharDataset(text, block_size=args.block_size)
    base_state = build_base_state(args, data, device)

    replacement_rows = []
    selected = []
    for optimizer in requested:
        candidate = tune_optimizer(optimizer, base_state, data, args, device)
        selected.append(candidate)
        rows = run_candidate(candidate, base_state, data, args, device, args.final_steps)
        replacement_rows.extend(rows)
        write_results(rows, args.work_dir / "finals" / candidate.optimizer)

    write_tune_summary(
        args.work_dir / "selected.csv",
        [(candidate, summarize_rows([row for row in replacement_rows if row["optimizer"] == candidate.optimizer])) for candidate in selected],
    )
    splice_results(args.results, replacement_rows)
    print(f"updated {args.results}")


if __name__ == "__main__":
    main()
