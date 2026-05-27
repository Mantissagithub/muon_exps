import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/muon_exps_matplotlib")

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "char_lm" / "results.csv"
OUT = ROOT / "artifacts" / "char_lm" / "loss_curves.png"

DISPLAY_NAMES = {
    "adamw": "AdamW",
    "torch_muon": "Torch Muon",
    "muon_like": "MuonLike",
    "normuon": "NorMuon",
    "u_normuon": "U-NorMuon",
    "aurora": "Aurora",
    "riemann_aurora": "Riemann Aurora",
}

PALETTE = {
    "adamw": "#111827",
    "torch_muon": "#2563eb",
    "muon_like": "#0f766e",
    "normuon": "#dc2626",
    "u_normuon": "#ea580c",
    "aurora": "#7c3aed",
    "riemann_aurora": "#9333ea",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def load_results(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))

    by_opt = {}
    for row in rows:
        parsed = {
            "optimizer": row["optimizer"],
            "step": int(row["step"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "tokens_per_sec": float(row["tokens_per_sec"]),
            "elapsed_s": float(row["elapsed_s"]),
            "best_val_loss": float(row["best_val_loss"]),
        }
        by_opt.setdefault(row["optimizer"], []).append(parsed)

    for opt_rows in by_opt.values():
        opt_rows.sort(key=lambda row: row["step"])
    return by_opt


def summarize(by_opt):
    summaries = []
    for opt, rows in by_opt.items():
        best_idx, best_row = min(enumerate(rows), key=lambda item: item[1]["val_loss"])
        final_row = rows[-1]
        summaries.append({
            "optimizer": opt,
            "label": DISPLAY_NAMES.get(opt, opt),
            "rows": rows,
            "best_idx": best_idx,
            "best_step": best_row["step"],
            "best_val": best_row["val_loss"],
            "final_val": final_row["val_loss"],
            "final_elapsed_s": final_row["elapsed_s"],
        })
    summaries.sort(key=lambda item: item["best_val"])
    return summaries


def spread_positions(values, gap):
    placed = []
    for value in values:
        if not placed:
            placed.append(value)
            continue
        placed.append(max(value, placed[-1] + gap))

    for idx in range(len(placed) - 2, -1, -1):
        placed[idx] = min(placed[idx], placed[idx + 1] - gap)
    return placed


def main():
    args = parse_args()
    by_opt = load_results(args.results)
    if not by_opt:
        raise ValueError(f"no rows found in {args.results}")

    summaries = summarize(by_opt)
    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    all_val_losses = [row["val_loss"] for summary in summaries for row in summary["rows"]]
    ymin = min(all_val_losses)
    ymax = max(all_val_losses)
    yrange = max(ymax - ymin, 1e-6)
    label_gap = max(0.018, yrange * 0.04)

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3.3, 1.4], height_ratios=[1, 1], wspace=0.22, hspace=0.28)
    ax_curve = fig.add_subplot(gs[:, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[1, 1])

    for side_ax in [ax_curve, ax_rank, ax_time]:
        side_ax.set_facecolor("#fbfbfc")
        for spine in side_ax.spines.values():
            spine.set_color("#d4d4d8")

    ax_curve.grid(axis="y", color="#e5e7eb", linewidth=1.0)
    ax_curve.grid(axis="x", color="#f1f5f9", linewidth=0.7)

    ordered_for_labels = sorted(summaries, key=lambda item: item["final_val"])
    final_targets = spread_positions([item["final_val"] for item in ordered_for_labels], label_gap)
    final_label_map = {
        item["optimizer"]: target
        for item, target in zip(ordered_for_labels, final_targets)
    }

    label_x = max_step * 1.065
    for summary in summaries:
        opt = summary["optimizer"]
        rows = summary["rows"]
        color = PALETTE.get(opt, "#334155")
        steps = [row["step"] for row in rows]
        vals = [row["val_loss"] for row in rows]
        best_idx = summary["best_idx"]
        markevery = max(1, len(steps) // 8)

        ax_curve.plot(
            steps,
            vals,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            markevery=markevery,
            solid_capstyle="round",
        )
        ax_curve.scatter(
            [steps[best_idx]],
            [vals[best_idx]],
            s=54,
            facecolors="white",
            edgecolors=color,
            linewidths=2,
            zorder=5,
        )

        final_x = steps[-1]
        final_y = vals[-1]
        label_y = final_label_map[opt]
        ax_curve.plot([final_x, label_x - max_step * 0.01], [final_y, label_y], color=color, linewidth=1.1, alpha=0.8)
        ax_curve.text(
            label_x,
            label_y,
            f"{summary['label']}  {final_y:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color=color,
            fontweight="semibold",
        )

    leader = summaries[0]
    runner_up = summaries[1] if len(summaries) > 1 else None
    lead_text = f"best val: {leader['label']} at {leader['best_val']:.3f}"
    if runner_up is not None:
        gap = runner_up["best_val"] - leader["best_val"]
        lead_text += f"   |   margin to next: {gap:.3f}"

    ax_curve.set_xlabel("training step", fontsize=11, color="#374151")
    ax_curve.set_ylabel("validation loss", fontsize=11, color="#374151")
    ax_curve.tick_params(colors="#4b5563", labelsize=10)
    ax_curve.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax_curve.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax_curve.set_xlim(0, max_step * 1.28)
    ax_curve.set_ylim(ymin - yrange * 0.05, max(ymax + yrange * 0.06, max(final_targets) + yrange * 0.08))

    rank_labels = [summary["label"] for summary in summaries]
    rank_values = [summary["best_val"] for summary in summaries]
    rank_colors = [PALETTE.get(summary["optimizer"], "#334155") for summary in summaries]
    y_rank = list(range(len(summaries)))

    ax_rank.barh(y_rank, rank_values, color=rank_colors, height=0.62)
    ax_rank.invert_yaxis()
    ax_rank.set_title("best validation loss", loc="left", fontsize=13, fontweight="bold", color="#111827", pad=10)
    ax_rank.set_yticks(y_rank, rank_labels)
    ax_rank.tick_params(axis="y", labelsize=10, colors="#374151")
    ax_rank.tick_params(axis="x", labelsize=9, colors="#6b7280")
    ax_rank.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax_rank.grid(axis="x", color="#e5e7eb", linewidth=1.0)
    ax_rank.set_axisbelow(True)
    ax_rank.set_xlim(0, max(rank_values) * 1.12)
    for idx, value in enumerate(rank_values):
        ax_rank.text(value + 0.002, idx, f"{value:.3f}", va="center", ha="left", fontsize=9.5, color="#111827")

    time_sorted = sorted(summaries, key=lambda item: item["final_elapsed_s"])
    time_labels = [summary["label"] for summary in time_sorted]
    time_values = [summary["final_elapsed_s"] for summary in time_sorted]
    time_colors = [PALETTE.get(summary["optimizer"], "#334155") for summary in time_sorted]
    y_time = list(range(len(time_sorted)))

    ax_time.barh(y_time, time_values, color=time_colors, height=0.62)
    ax_time.invert_yaxis()
    ax_time.set_title("wall time to final checkpoint", loc="left", fontsize=13, fontweight="bold", color="#111827", pad=10)
    ax_time.set_yticks(y_time, time_labels)
    ax_time.tick_params(axis="y", labelsize=10, colors="#374151")
    ax_time.tick_params(axis="x", labelsize=9, colors="#6b7280")
    ax_time.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}s"))
    ax_time.grid(axis="x", color="#e5e7eb", linewidth=1.0)
    ax_time.set_axisbelow(True)
    ax_time.set_xlim(0, max(time_values) * 1.16)
    for idx, value in enumerate(time_values):
        ax_time.text(value + max(time_values) * 0.015, idx, f"{value:.0f}s", va="center", ha="left", fontsize=9.5, color="#111827")

    fig.suptitle("TinyShakespeare validation loss", x=0.08, y=0.972, ha="left", fontsize=20, fontweight="bold", color="#111827")
    fig.text(
        0.08,
        0.92,
        f"lower is better · {max_step:,} training steps · {len(summaries)} optimizers · {lead_text}",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#4b5563",
    )
    fig.subplots_adjust(top=0.905, right=0.96, left=0.08, bottom=0.09)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
