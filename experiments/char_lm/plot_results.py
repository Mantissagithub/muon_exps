import argparse
import csv
import html
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/muon_exps_matplotlib")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import ticker
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "char_lm" / "results.csv"
OUT = ROOT / "artifacts" / "char_lm" / "loss_curves.png"
HTML_OUT = ROOT / "artifacts" / "char_lm" / "loss_report.html"
BETA_OUT = ROOT / "artifacts" / "char_lm" / "beta_schedule.png"
DISTANCE_OUT = ROOT / "artifacts" / "char_lm" / "amuse_distances.png"
COSINE_OUT = ROOT / "artifacts" / "char_lm" / "update_cosine.png"

DISPLAY_NAMES = {
    "adamw": "AdamW",
    "torch_muon": "Muon",
    "muon_like": "MuonLike",
    "normuon": "NorMuon",
    "u_normuon": "U-NorMuon",
    "aurora": "Aurora",
    "riemann_aurora": "Riemann Aurora",
    "adafactor": "Adafactor",
    "sf_adamw": "SF-AdamW",
    "amuse_muon": "AMUSE-Muon",
    "sf_muon_fixed_beta_0.6": "SF-Muon beta=0.6",
    "sf_muon_fixed_beta_0.9": "SF-Muon beta=0.9",
    "lion": "Lion",
    "adopt": "ADOPT",
    "prodigy": "Prodigy",
    "soap": "SOAP",
    "mars": "MARS",
    "sophia": "Sophia",
    "ademamix": "AdEMAMix",
}

PALETTE = {
    "adamw": "#1f2937",
    "torch_muon": "#2563eb",
    "muon_like": "#0f766e",
    "normuon": "#dc2626",
    "u_normuon": "#c2410c",
    "aurora": "#7c3aed",
    "riemann_aurora": "#9333ea",
    "adafactor": "#6b7280",
    "sf_adamw": "#16a34a",
    "amuse_muon": "#0f9d7a",
    "sf_muon_fixed_beta_0.6": "#84a11d",
    "sf_muon_fixed_beta_0.9": "#b45309",
    "lion": "#ca8a04",
    "adopt": "#b91c1c",
    "prodigy": "#0891b2",
    "soap": "#111827",
    "mars": "#a855f7",
    "sophia": "#4f46e5",
    "ademamix": "#9333ea",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--html-out", type=Path, default=HTML_OUT)
    p.add_argument("--beta-out", type=Path, default=BETA_OUT)
    p.add_argument("--distance-out", type=Path, default=DISTANCE_OUT)
    p.add_argument("--cosine-out", type=Path, default=COSINE_OUT)
    return p.parse_args()


def parse_float(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


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
            "val_loss_z": parse_float(row, "val_loss_z"),
            "beta_t": parse_float(row, "beta_t"),
            "lr": parse_float(row, "lr"),
            "step_time": parse_float(row, "step_time"),
            "tokens_per_sec": float(row["tokens_per_sec"]),
            "elapsed_s": float(row["elapsed_s"]),
            "best_val_loss": float(row["best_val_loss"]),
            "update_cosine_similarity": parse_float(row, "update_cosine_similarity"),
            "z_x_distance": parse_float(row, "z_x_distance"),
            "y_x_distance": parse_float(row, "y_x_distance"),
            "y_z_distance": parse_float(row, "y_z_distance"),
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


def fmt_step(value):
    return f"{value:,}"


def fmt_loss(value):
    return f"{value:.3f}"


def fmt_seconds(value):
    if value >= 60:
        minutes = int(value // 60)
        seconds = int(round(value - minutes * 60))
        return f"{minutes}m {seconds:02d}s"
    return f"{value:.1f}s"


def build_svg_path(rows, x0, y0, width, height, xmin, xmax, ymin, ymax):
    coords = []
    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)
    for row in rows:
        x = x0 + width * ((row["step"] - xmin) / xr)
        y = y0 + height * (1.0 - ((row["val_loss"] - ymin) / yr))
        coords.append((x, y))
    if not coords:
        return "", []
    parts = [f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"]
    for x, y in coords[1:]:
        parts.append(f"L {x:.2f} {y:.2f}")
    return " ".join(parts), coords


def render_html_report(args, summaries):
    leader = summaries[0]
    runner_up = summaries[1] if len(summaries) > 1 else None
    image_name = html.escape(args.out.name)
    beta_name = html.escape(args.beta_out.name)
    dist_name = html.escape(args.distance_out.name)
    cosine_name = html.escape(args.cosine_out.name)

    rows = []
    for summary in summaries:
        final = summary["rows"][-1]
        rows.append(
            f"""
            <tr>
              <td>{html.escape(summary['label'])}</td>
              <td>{fmt_loss(summary['best_val'])}</td>
              <td>{fmt_loss(summary['final_val'])}</td>
              <td>{fmt_step(summary['best_step'])}</td>
              <td>{fmt_seconds(summary['final_elapsed_s'])}</td>
              <td>{int(final['tokens_per_sec']):,}</td>
            </tr>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TinyShakespeare optimizer report</title>
  <style>
    body {{
      margin: 0;
      background: #ffffff;
      color: #111111;
      font-family: "DejaVu Serif", "Times New Roman", serif;
    }}
    main {{
      width: min(1120px, calc(100vw - 40px));
      margin: 28px auto 40px;
    }}
    h1 {{
      font-size: 24px;
      font-weight: 400;
      margin: 0 0 6px;
    }}
    .deck {{
      color: #333333;
      font-size: 14px;
      line-height: 1.5;
      margin-bottom: 18px;
    }}
    .figure {{
      margin: 18px 0 26px;
    }}
    .figure img {{
      width: 100%;
      display: block;
      border: 1px solid #d5d5d5;
    }}
    .caption {{
      margin-top: 8px;
      font-size: 13px;
      color: #333333;
      line-height: 1.45;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 20px 0;
    }}
    .metric {{
      border: 1px solid #d5d5d5;
      padding: 10px 12px;
    }}
    .metric .k {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #666666;
    }}
    .metric .v {{
      margin-top: 6px;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #dcdcdc;
      padding: 8px 10px;
      text-align: left;
    }}
    th {{
      font-weight: 600;
      background: #f7f7f7;
    }}
    .diag-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .diag-grid img {{
      width: 100%;
      display: block;
      border: 1px solid #d5d5d5;
    }}
    @media (max-width: 860px) {{
      .metrics, .diag-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>TinyShakespeare optimizer report</h1>
    <div class="deck">
      Same initialization, same train/validation split, and same seed across all compared optimizers.
      The main figure emphasizes validation-loss trajectories; auxiliary figures isolate the AMUSE schedule and sequence diagnostics.
    </div>

    <div class="metrics">
      <div class="metric">
        <div class="k">Best run</div>
        <div class="v">{html.escape(leader['label'])} · {fmt_loss(leader['best_val'])}</div>
      </div>
      <div class="metric">
        <div class="k">Best step</div>
        <div class="v">{fmt_step(leader['best_step'])}</div>
      </div>
      <div class="metric">
        <div class="k">Margin to next</div>
        <div class="v">{fmt_loss((runner_up['best_val'] - leader['best_val']) if runner_up else 0.0)}</div>
      </div>
    </div>

    <section class="figure">
      <img src="{image_name}" alt="Validation loss comparison">
      <div class="caption">
        <strong>Figure 1.</strong> Validation loss against training step for the benchmarked optimizers.
        The right panel zooms into the late-training region to expose the final spread between methods.
      </div>
    </section>

    <table>
      <thead>
        <tr>
          <th>Optimizer</th>
          <th>Best val</th>
          <th>Final val</th>
          <th>Best step</th>
          <th>Wall time</th>
          <th>Tokens/sec</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>

    <section class="diag-grid">
      <div class="figure">
        <img src="{beta_name}" alt="AMUSE beta schedule">
        <div class="caption"><strong>Figure 2.</strong> The AMUSE schedule parameter $\\beta_t$ over training.</div>
      </div>
      <div class="figure">
        <img src="{dist_name}" alt="AMUSE sequence distances">
        <div class="caption"><strong>Figure 3.</strong> Relative distances between the x, y, and z AMUSE sequences.</div>
      </div>
      <div class="figure">
        <img src="{cosine_name}" alt="AMUSE update cosine similarity">
        <div class="caption"><strong>Figure 4.</strong> Cosine similarity between consecutive AMUSE x-sequence updates.</div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(html_text, encoding="utf-8")


def dashboard_bounds(summaries):
    vals = [row["val_loss"] for summary in summaries for row in summary["rows"]]
    ymin = min(vals)
    ymax = max(vals)
    span = max(ymax - ymin, 1e-6)
    return ymin - span * 0.04, ymax + span * 0.08


def add_card(ax, xy, width, height, edge="#223047", face="#07101e", lw=1.0, alpha=0.98):
    card = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012",
        transform=ax.transAxes,
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
        clip_on=False,
    )
    ax.add_patch(card)
    return card


def dashboard_color(optimizer):
    if optimizer == "adamw":
        return "#e5e7eb"
    return PALETTE.get(optimizer, "#94a3b8")


def setup_dashboard_axes(fig):
    fig.patch.set_facecolor("#050914")
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_axis_off()
    ax_bg.set_facecolor("#050914")
    ax_bg.add_patch(
        patches.Rectangle((0, 0), 1, 1, transform=ax_bg.transAxes, facecolor="#050914", edgecolor="none")
    )
    ax_bg.add_patch(
        patches.Circle((0.33, 0.80), 0.42, transform=ax_bg.transAxes, facecolor="#0b2b55", edgecolor="none", alpha=0.22)
    )
    ax_bg.add_patch(
        patches.Circle((0.76, 0.18), 0.36, transform=ax_bg.transAxes, facecolor="#1c0f39", edgecolor="none", alpha=0.18)
    )

    ax_chart = fig.add_axes([0.055, 0.25, 0.665, 0.60])
    ax_rank = fig.add_axes([0.735, 0.25, 0.235, 0.60])
    ax_bottom = fig.add_axes([0.055, 0.055, 0.915, 0.13])
    for ax in [ax_chart, ax_rank, ax_bottom]:
        ax.set_facecolor("#07101e")
        for spine in ax.spines.values():
            spine.set_color("#223047")
            spine.set_linewidth(1.0)
    return ax_bg, ax_chart, ax_rank, ax_bottom


def style_dashboard_chart(ax, summaries):
    ymin, ymax = dashboard_bounds(summaries)
    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    ax.set_xlim(0, max_step)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color="#1e2a3f", linestyle="--", linewidth=0.8, alpha=0.85)
    ax.tick_params(colors="#a8b3c7", labelsize=10)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("training step", color="#c7d2e8", fontsize=11)
    ax.set_ylabel("")
    ax.text(
        0.02,
        0.96,
        "Validation loss (lower is better)",
        transform=ax.transAxes,
        color="#aeb8cc",
        fontsize=11,
        fontweight="semibold",
        va="top",
    )
    return ymin, ymax, max_step


def draw_dashboard_header(fig, summaries):
    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    fig.text(0.078, 0.918, "TinyShakespeare validation loss", color="#f8fafc", fontsize=25, fontweight="bold")
    fig.text(
        0.078,
        0.887,
        f"{fmt_step(max_step)} steps  ·  {len(summaries)} optimizers  ·  lower is better",
        color="#8f9bae",
        fontsize=11,
    )


def plot_dashboard_lines(ax, summaries, active_points=None, show_labels=True):
    ymin, ymax, max_step = style_dashboard_chart(ax, summaries)
    line_artists = {}
    point_artists = {}
    active_points = active_points or max(len(summary["rows"]) for summary in summaries)

    for summary in summaries:
        color = dashboard_color(summary["optimizer"])
        rows = summary["rows"][:active_points]
        xs = [row["step"] for row in rows]
        ys = [row["val_loss"] for row in rows]
        (line,) = ax.plot(xs, ys, color=color, linewidth=2.0, solid_capstyle="round")
        point = ax.scatter(xs[-1:], ys[-1:], s=34, color=color, edgecolors="#dbeafe", linewidths=1.0, zorder=6)
        line_artists[summary["optimizer"]] = line
        point_artists[summary["optimizer"]] = point

    leader = summaries[0]
    if show_labels:
        best_step = leader["best_step"]
        ax.axvline(best_step, color="#d1d5db", linestyle="--", linewidth=1.1, alpha=0.75)
        ax.text(
            best_step,
            ymax - (ymax - ymin) * 0.055,
            fmt_step(best_step),
            color="#ff6b00",
            fontsize=10,
            fontweight="bold",
            ha="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#07101e", "edgecolor": "#223047"},
        )
        final_rows = sorted(summaries, key=lambda item: item["best_val"], reverse=True)
        top = min(ymax - (ymax - ymin) * 0.18, 2.25)
        bottom = max(ymin + (ymax - ymin) * 0.10, 1.54)
        label_values = [top - idx * ((top - bottom) / max(len(final_rows) - 1, 1)) for idx in range(len(final_rows))]
        x_label = max_step * 0.80
        for summary, label_y in zip(final_rows, label_values):
            color = dashboard_color(summary["optimizer"])
            ax.text(
                x_label,
                label_y,
                f"{summary['label']}  {summary['best_val']:.3f}",
                color=color,
                fontsize=10,
                fontweight="semibold",
                va="center",
            )
    return line_artists, point_artists


def draw_dashboard_rank(ax, summaries, active_points=None):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_color("#223047")

    active_points = active_points or max(len(summary["rows"]) for summary in summaries)
    leader = summaries[0]
    runner_up = summaries[1] if len(summaries) > 1 else None
    gap = runner_up["best_val"] - leader["best_val"] if runner_up else 0.0

    add_card(ax, (0.03, 0.76), 0.94, 0.20, edge="#ff6b00", face="#0a0f1c", lw=1.1)
    ax.text(0.08, 0.92, "CURRENT LEADER", color="#ff6b00", fontsize=8.8, fontweight="bold", va="top")
    ax.text(0.08, 0.85, leader["label"], color="#f8fafc", fontsize=18, fontweight="bold", va="top")
    ax.text(0.90, 0.80, f"{leader['best_val']:.3f}", color="#ff6b00", fontsize=24, fontweight="bold", ha="right", va="top")
    ax.text(0.08, 0.78, f"+{gap:.3f} vs {runner_up['label'] if runner_up else 'next'}", color="#ff6b00", fontsize=9.5, fontweight="bold")

    progress = (active_points - 1) / max(max(len(summary["rows"]) for summary in summaries) - 1, 1)
    current_step = int(round(max(summary["rows"][-1]["step"] for summary in summaries) * progress))
    add_card(ax, (0.03, 0.58), 0.94, 0.14, edge="#223047", face="#091426", lw=1.0)
    ax.text(0.08, 0.69, "PROGRESS", color="#9aa6bb", fontsize=8.5, fontweight="bold", va="top")
    ax.text(0.08, 0.62, f"step {fmt_step(current_step)}", color="#f8fafc", fontsize=16, fontweight="bold")
    ax.text(0.52, 0.62, f"/ {fmt_step(max(summary['rows'][-1]['step'] for summary in summaries))}", color="#9aa6bb", fontsize=11)
    ax.add_patch(patches.FancyBboxPatch((0.08, 0.595), 0.82, 0.012, boxstyle="round,pad=0", transform=ax.transAxes, facecolor="#2b3342", edgecolor="none"))
    ax.add_patch(patches.FancyBboxPatch((0.08, 0.595), 0.82 * progress, 0.012, boxstyle="round,pad=0", transform=ax.transAxes, facecolor="#ff6b00", edgecolor="none"))

    add_card(ax, (0.03, 0.04), 0.94, 0.50, edge="#223047", face="#091426", lw=1.0)
    ax.text(0.08, 0.50, "LIVE RANKING", color="#9aa6bb", fontsize=8.5, fontweight="bold", va="top")
    live_rows = []
    for summary in summaries:
        row = summary["rows"][min(active_points - 1, len(summary["rows"]) - 1)]
        live_rows.append((row["val_loss"], summary))
    live_rows.sort(key=lambda item: item[0])

    y = 0.44
    for idx, (current_val, summary) in enumerate(live_rows, start=1):
        color = dashboard_color(summary["optimizer"])
        if idx == 1:
            ax.add_patch(patches.Rectangle((0.06, y - 0.025), 0.88, 0.045, transform=ax.transAxes, facecolor="#261b13", edgecolor="none"))
        ax.text(0.08, y, str(idx), color="#cbd5e1", fontsize=9, va="center")
        ax.scatter([0.17], [y], s=28, color=color, transform=ax.transAxes)
        ax.text(0.22, y, summary["label"], color="#e5e7eb", fontsize=9.5, fontweight="semibold", va="center")
        ax.text(0.90, y, f"{summary['best_val']:.3f}", color=color, fontsize=10, fontweight="bold", ha="right", va="center")
        y -= 0.061


def draw_dashboard_bottom(ax, summaries):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_color("#223047")

    leader = summaries[0]
    fastest = sorted(summaries, key=lambda item: item["final_elapsed_s"])[:5]
    ax.text(0.04, 0.65, "WINNER", color="#ff6b00", fontsize=9, fontweight="bold")
    ax.text(0.04, 0.34, leader["label"], color="#f8fafc", fontsize=19, fontweight="bold")
    ax.text(0.20, 0.34, f"{leader['best_val']:.3f}", color="#ff6b00", fontsize=23, fontweight="bold")
    ax.plot([0.30, 0.30], [0.18, 0.82], color="#223047", linewidth=1)
    ax.text(0.33, 0.70, "WALL TIME TO FINAL CHECKPOINT", color="#9aa6bb", fontsize=9, fontweight="bold")

    x = 0.33
    max_elapsed = max(summary["final_elapsed_s"] for summary in fastest)
    for summary in fastest[:4]:
        color = dashboard_color(summary["optimizer"])
        ax.text(x, 0.48, summary["label"], color="#e5e7eb", fontsize=9, fontweight="semibold")
        ax.add_patch(patches.FancyBboxPatch((x, 0.28), 0.07, 0.055, boxstyle="round,pad=0.01", transform=ax.transAxes, facecolor="#2b3342", edgecolor="none"))
        ax.add_patch(patches.FancyBboxPatch((x, 0.28), 0.07 * (summary["final_elapsed_s"] / max_elapsed), 0.055, boxstyle="round,pad=0.01", transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(x + 0.08, 0.30, fmt_seconds(summary["final_elapsed_s"]), color="#cbd5e1", fontsize=9, va="center")
        x += 0.16


def render_dashboard_png(args, summaries):
    fig = plt.figure(figsize=(14.6, 8.2), dpi=180)
    _, ax_chart, ax_rank, ax_bottom = setup_dashboard_axes(fig)
    draw_dashboard_header(fig, summaries)
    plot_dashboard_lines(ax_chart, summaries, show_labels=True)
    draw_dashboard_rank(ax_rank, summaries)
    draw_dashboard_bottom(ax_bottom, summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_clean_png(args, summaries):
    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    all_losses = [row["val_loss"] for summary in summaries for row in summary["rows"]]
    ymin = min(all_losses)
    ymax = max(all_losses)
    span = max(ymax - ymin, 1e-6)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    })
    fig = plt.figure(figsize=(12.4, 4.2), dpi=220, facecolor="white")
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.22)
    ax_all = fig.add_subplot(grid[0, 0])
    ax_focus = fig.add_subplot(grid[0, 1])

    ordered = sorted(summaries, key=lambda item: item["best_val"])
    available = {summary["optimizer"]: summary for summary in ordered}
    focus_keys = []
    for name in ["adamw", "torch_muon", "muon_like", "sf_adamw", "amuse_muon"]:
        if name == "muon_like" and "torch_muon" in available:
            continue
        if name in available:
            focus_keys.append(name)
    focus = [available[name] for name in focus_keys] if focus_keys else ordered[: min(4, len(ordered))]

    for axis in (ax_all, ax_focus):
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.8)
        axis.grid(True, color="#b0b0b0", linewidth=0.6, alpha=0.6)
        axis.tick_params(colors="#222222", labelsize=8, width=0.6, length=3)
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0"))
        axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        axis.set_xlabel("Step", fontsize=9, color="#222222")
        axis.set_ylabel("Validation Perplexity", fontsize=9, color="#222222")

    ax_all.set_title("FineWeb (Llama 124M)", fontsize=12, pad=6)
    ax_focus.set_title("FineWeb (Llama 124M)", fontsize=12, pad=6)
    ax_all.set_xlim(0, max_step)
    ax_all.set_ylim(max(0.0, ymin - span * 0.04), ymax + span * 0.03)

    focus_steps = []
    focus_vals = []
    for summary in ordered:
        steps = [row["step"] for row in summary["rows"]]
        vals = [row["val_loss"] for row in summary["rows"]]
        ax_all.plot(
            steps,
            vals,
            color=PALETTE.get(summary["optimizer"], "#334155"),
            linewidth=1.15,
            alpha=0.95,
            label=summary["label"],
        )

    for summary in focus:
        steps = [row["step"] for row in summary["rows"]]
        vals = [row["val_loss"] for row in summary["rows"]]
        focus_steps.extend(steps)
        focus_vals.extend(vals)
        ax_focus.plot(
            steps,
            vals,
            color=PALETTE.get(summary["optimizer"], "#334155"),
            linewidth=1.2,
            alpha=0.95,
            label=summary["label"],
        )

    if focus_steps and focus_vals:
        ax_focus.set_xlim(min(focus_steps), max(focus_steps) * 1.02)
        fmin = min(focus_vals)
        fmax = max(focus_vals)
        fspan = max(fmax - fmin, 1e-6)
        ax_focus.set_ylim(fmin - fspan * 0.04, fmax + fspan * 0.04)

    legend_all = ax_all.legend(loc="upper right", frameon=True, fontsize=7, ncol=2)
    legend_all.get_frame().set_edgecolor("#cccccc")
    legend_all.get_frame().set_facecolor("white")
    legend_all.get_frame().set_alpha(0.95)
    legend_focus = ax_focus.legend(loc="upper right", frameon=True, fontsize=7)
    legend_focus.get_frame().set_edgecolor("#cccccc")
    legend_focus.get_frame().set_facecolor("white")
    legend_focus.get_frame().set_alpha(0.95)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def has_metric(summaries, key):
    return any(math.isfinite(row.get(key, float("nan"))) for summary in summaries for row in summary["rows"])


def render_metric_png(args, summaries, out_path, keys, title, ylabel):
    if not any(has_metric(summaries, key) for key in keys):
        return False

    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220, facecolor="white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.55)
    ax.tick_params(colors="#222222", labelsize=8, width=0.6, length=3)
    ax.set_xlabel("step", color="#222222", fontsize=8, labelpad=6)
    ax.set_ylabel(ylabel, color="#222222", fontsize=8, labelpad=6)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0k"))
    ax.set_xlim(0, max_step)

    labels = {
        "beta_t": "beta_t",
        "z_x_distance": "||z - x|| / ||x||",
        "y_x_distance": "||y - x|| / ||x||",
        "y_z_distance": "||y - z|| / ||z||",
        "update_cosine_similarity": "cos(delta x_t, delta x_{t-1})",
    }
    line_idx = 0
    for summary in summaries:
        base_color = PALETTE.get(summary["optimizer"], "#334155")
        for key in keys:
            rows = [row for row in summary["rows"] if math.isfinite(row.get(key, float("nan")))]
            if not rows:
                continue
            xs = [row["step"] for row in rows]
            ys = [row[key] for row in rows]
            label = f"{summary['label']} {labels.get(key, key)}" if len(keys) > 1 else summary["label"]
            linestyle = ["-", "--", ":"][line_idx % 3] if len(keys) > 1 else "-"
            ax.plot(xs, ys, color=base_color, linewidth=1.25, alpha=0.95, linestyle=linestyle, label=label)
            line_idx += 1

    legend = ax.legend(loc="best", frameon=False, fontsize=7, handlelength=2.2)
    for text in legend.get_texts():
        text.set_color("#222222")

    fig.text(0.09, 0.95, title, color="#111111", fontsize=9.8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    by_opt = load_results(args.results)
    if not by_opt:
        raise ValueError(f"no rows found in {args.results}")

    summaries = summarize(by_opt)
    render_clean_png(args, summaries)
    render_html_report(args, summaries)
    wrote_beta = render_metric_png(args, summaries, args.beta_out, ["beta_t"], "AMUSE beta schedule", "beta")
    wrote_distance = render_metric_png(
        args,
        summaries,
        args.distance_out,
        ["z_x_distance", "y_x_distance", "y_z_distance"],
        "AMUSE sequence distances",
        "relative distance",
    )
    wrote_cosine = render_metric_png(
        args,
        summaries,
        args.cosine_out,
        ["update_cosine_similarity"],
        "AMUSE x-update cosine similarity",
        "cosine similarity",
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.html_out}")
    if wrote_beta:
        print(f"wrote {args.beta_out}")
    if wrote_distance:
        print(f"wrote {args.distance_out}")
    if wrote_cosine:
        print(f"wrote {args.cosine_out}")


if __name__ == "__main__":
    main()
