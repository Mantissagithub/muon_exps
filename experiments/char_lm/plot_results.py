import csv
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/muon_exps_matplotlib")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "char_lm" / "results.csv"
OUT = ROOT / "artifacts" / "char_lm" / "loss_curves.png"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def main():
    args = parse_args()
    with args.results.open() as f:
        rows = list(csv.DictReader(f))

    by_opt = {}
    for row in rows:
        by_opt.setdefault(row["optimizer"], []).append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for opt, opt_rows in by_opt.items():
        steps = [int(row["step"]) for row in opt_rows]
        vals = [float(row["val_loss"]) for row in opt_rows]
        plt.plot(steps, vals, marker="o", label=opt)
    plt.xlabel("step")
    plt.ylabel("validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
