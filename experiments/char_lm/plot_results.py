import argparse
import csv
import html
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/muon_exps_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import ticker
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "char_lm" / "results.csv"
OUT = ROOT / "artifacts" / "char_lm" / "loss_curves.png"
HTML_OUT = ROOT / "artifacts" / "char_lm" / "loss_report.html"
GIF_OUT = ROOT / "artifacts" / "char_lm" / "loss_curves.gif"

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
    p.add_argument("--html-out", type=Path, default=HTML_OUT)
    p.add_argument("--gif-out", type=Path, default=GIF_OUT)
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
    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    all_val_losses = [row["val_loss"] for summary in summaries for row in summary["rows"]]
    ymin = min(all_val_losses)
    ymax = max(all_val_losses)
    yrange = max(ymax - ymin, 1e-9)
    ypad = yrange * 0.05
    ymin -= ypad
    ymax += ypad

    chart_w = 920
    chart_h = 500
    chart_left = 74
    chart_top = 28
    plot_w = 720
    plot_h = 404
    chart_right = chart_left + plot_w
    chart_bottom = chart_top + plot_h

    leader = summaries[0]
    runner_up = summaries[1] if len(summaries) > 1 else None
    gap = runner_up["best_val"] - leader["best_val"] if runner_up else 0.0
    fastest = min(summaries, key=lambda item: item["final_elapsed_s"])
    sharpest_finish = min(summaries, key=lambda item: item["final_val"])
    image_name = html.escape(args.out.name)

    grid_lines = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_top + plot_h * frac
        loss = ymax - (y - chart_top) / plot_h * (ymax - ymin)
        grid_lines.append(
            f'<line x1="{chart_left}" y1="{y:.2f}" x2="{chart_right}" y2="{y:.2f}" class="grid-line"/>'
            f'<text x="{chart_left - 14}" y="{y + 4:.2f}" class="axis-label axis-label-y">{fmt_loss(loss)}</text>'
        )

    x_ticks = []
    tick_count = 6
    for idx in range(tick_count + 1):
        step = int(round(max_step * idx / tick_count))
        x = chart_left + plot_w * idx / tick_count
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{chart_top}" x2="{x:.2f}" y2="{chart_bottom}" class="grid-line grid-line-v"/>'
            f'<text x="{x:.2f}" y="{chart_bottom + 28}" class="axis-label axis-label-x">{fmt_step(step)}</text>'
        )

    line_layers = []
    endpoint_labels = []
    ordered_for_labels = sorted(summaries, key=lambda item: item["final_val"])
    label_gap = max(18.0, plot_h * 0.042)
    label_targets = spread_positions(
        [
            chart_top + plot_h * (1.0 - ((item["final_val"] - ymin) / max(ymax - ymin, 1e-9)))
            for item in ordered_for_labels
        ],
        label_gap,
    )
    label_map = {
        item["optimizer"]: target
        for item, target in zip(ordered_for_labels, label_targets)
    }

    for idx, summary in enumerate(summaries):
        color = PALETTE.get(summary["optimizer"], "#334155")
        path_d, coords = build_svg_path(summary["rows"], chart_left, chart_top, plot_w, plot_h, 0, max_step, ymin, ymax)
        if not coords:
            continue
        best = summary["rows"][summary["best_idx"]]
        best_x = chart_left + plot_w * (best["step"] / max(max_step, 1))
        best_y = chart_top + plot_h * (1.0 - ((best["val_loss"] - ymin) / max(ymax - ymin, 1e-9)))
        final = summary["rows"][-1]
        final_x = coords[-1][0]
        final_y = coords[-1][1]
        label_y = label_map[summary["optimizer"]]
        delay = 0.08 * idx
        line_layers.append(
            f"""
            <g class="series" style="--series-color:{color}; --delay:{delay:.2f}s">
              <path d="{path_d}" class="series-line"/>
              <circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="6.4" class="best-ring"/>
              <circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="2.6" class="best-core"/>
              <path d="M {final_x:.2f} {final_y:.2f} L {chart_right + 18:.2f} {label_y:.2f}" class="label-link"/>
            </g>
            """
        )
        endpoint_labels.append(
            f"""
            <div class="endpoint-tag" style="top:{label_y - 14:.2f}px; color:{color}; border-color:{color}22; background:{color}10;">
              <span class="endpoint-name">{html.escape(summary['label'])}</span>
              <span class="endpoint-value">{fmt_loss(summary['final_val'])}</span>
            </div>
            """
        )

    rank_cards = []
    rank_span = max(item["best_val"] for item in summaries) - min(item["best_val"] for item in summaries)
    rank_span = max(rank_span, 1e-9)
    for idx, summary in enumerate(summaries, start=1):
        color = PALETTE.get(summary["optimizer"], "#334155")
        fill = 0.35 + 0.65 * (1.0 - ((summary["best_val"] - leader["best_val"]) / rank_span))
        rank_cards.append(
            f"""
            <div class="rank-row" style="--fill:{fill:.4f}; --rank-color:{color};">
              <div class="rank-head">
                <span class="rank-index">{idx:02d}</span>
                <span class="rank-name">{html.escape(summary['label'])}</span>
                <span class="rank-score">{fmt_loss(summary['best_val'])}</span>
              </div>
              <div class="rank-bar"><span></span></div>
              <div class="rank-meta">best @ step {fmt_step(summary['best_step'])}</div>
            </div>
            """
        )

    time_sorted = sorted(summaries, key=lambda item: item["final_elapsed_s"])
    max_elapsed = max(item["final_elapsed_s"] for item in time_sorted)
    time_cards = []
    for idx, summary in enumerate(time_sorted):
        color = PALETTE.get(summary["optimizer"], "#334155")
        fill = summary["final_elapsed_s"] / max(max_elapsed, 1e-9)
        time_cards.append(
            f"""
            <div class="time-row" style="--fill:{fill:.4f}; --rank-color:{color}; --delay:{0.08 * idx:.2f}s;">
              <div class="time-head">
                <span class="time-name">{html.escape(summary['label'])}</span>
                <span class="time-score">{fmt_seconds(summary['final_elapsed_s'])}</span>
              </div>
              <div class="time-bar"><span></span></div>
            </div>
            """
        )

    table_rows = []
    for summary in summaries:
        final = summary["rows"][-1]
        table_rows.append(
            f"""
            <tr>
              <td><span class="swatch" style="background:{PALETTE.get(summary['optimizer'], '#334155')}"></span>{html.escape(summary['label'])}</td>
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
    :root {{
      --bg: #f3f0e8;
      --panel: rgba(255, 252, 246, 0.86);
      --panel-strong: #fffdf8;
      --ink: #171717;
      --muted: #5f5b52;
      --hairline: rgba(23, 23, 23, 0.12);
      --shadow: 0 22px 60px rgba(54, 41, 21, 0.12);
      --display: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
      --body: "Avenir Next", "IBM Plex Sans", "Segoe UI", sans-serif;
      --mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--body);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(212, 160, 23, 0.18), transparent 34%),
        radial-gradient(circle at 86% 0%, rgba(37, 99, 235, 0.12), transparent 24%),
        linear-gradient(180deg, #f7f4ed 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.08;
      background-image:
        linear-gradient(rgba(0, 0, 0, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 0, 0, 0.04) 1px, transparent 1px);
      background-size: 18px 18px;
      mask-image: linear-gradient(180deg, black 20%, transparent 100%);
    }}
    .page {{
      width: min(1500px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 30px 0 42px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 0.9fr;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }}
    .kicker {{
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #8b6f47;
      margin-bottom: 12px;
    }}
    h1 {{
      font-family: var(--display);
      font-size: clamp(52px, 6.2vw, 84px);
      line-height: 0.93;
      margin: 0;
      letter-spacing: -0.04em;
      font-weight: 700;
    }}
    .subtitle {{
      margin-top: 14px;
      max-width: 60ch;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
    }}
    .hero-stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}
    .stat-card, .panel {{
      position: relative;
      overflow: hidden;
      border: 1px solid var(--hairline);
      background: var(--panel);
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow);
      border-radius: 26px;
    }}
    .stat-card {{
      padding: 18px 18px 16px;
      animation: rise-in 0.8s cubic-bezier(.2,.8,.2,1) both;
    }}
    .stat-label {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 11px;
    }}
    .stat-value {{
      margin-top: 10px;
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .stat-meta {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .dashboard {{
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.9fr);
      gap: 18px;
    }}
    .panel {{
      padding: 22px;
      animation: rise-in 0.85s cubic-bezier(.2,.8,.2,1) both;
    }}
    .panel h2 {{
      margin: 0;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: #7a7468;
      font-weight: 600;
    }}
    .panel-title {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 16px;
    }}
    .panel-strong-label {{
      font-size: 28px;
      font-family: var(--display);
      letter-spacing: -0.03em;
    }}
    .panel-note {{
      color: var(--muted);
      font-size: 13px;
      font-family: var(--mono);
    }}
    .chart-frame {{
      position: relative;
      min-height: 560px;
      border-radius: 22px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.38)),
        linear-gradient(135deg, rgba(255,255,255,0.62), rgba(230,221,205,0.28));
      border: 1px solid rgba(23, 23, 23, 0.08);
      overflow: hidden;
    }}
    .chart-caption {{
      position: absolute;
      left: 24px;
      top: 18px;
      z-index: 2;
      max-width: 360px;
    }}
    .chart-caption .eyebrow {{
      font-size: 11px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #7f7464;
    }}
    .chart-caption h3 {{
      margin: 8px 0 6px;
      font-size: 30px;
      line-height: 1.0;
      font-family: var(--display);
      letter-spacing: -0.04em;
    }}
    .chart-caption p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .chart-svg {{
      width: 100%;
      height: 560px;
      display: block;
    }}
    .grid-line {{
      stroke: rgba(23, 23, 23, 0.08);
      stroke-width: 1;
    }}
    .grid-line-v {{
      stroke: rgba(23, 23, 23, 0.05);
    }}
    .axis-label {{
      fill: #726c61;
      font-family: var(--mono);
      font-size: 12px;
    }}
    .axis-label-y {{ text-anchor: end; }}
    .axis-label-x {{ text-anchor: middle; }}
    .series-line {{
      fill: none;
      stroke: var(--series-color);
      stroke-width: 3.2;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-dasharray: 2000;
      stroke-dashoffset: 2000;
      animation: draw-line 2.1s cubic-bezier(.2,.7,.15,1) forwards;
      animation-delay: var(--delay);
    }}
    .best-ring {{
      fill: rgba(255,255,255,0.94);
      stroke: var(--series-color);
      stroke-width: 2.4;
      opacity: 0;
      animation: pop-in 0.5s ease forwards;
      animation-delay: calc(var(--delay) + 1.3s);
    }}
    .best-core {{
      fill: var(--series-color);
      opacity: 0;
      animation: pop-in 0.5s ease forwards;
      animation-delay: calc(var(--delay) + 1.38s);
    }}
    .label-link {{
      fill: none;
      stroke: var(--series-color);
      stroke-width: 1.4;
      stroke-linecap: round;
      opacity: 0;
      animation: fade-in 0.45s ease forwards;
      animation-delay: calc(var(--delay) + 1.45s);
    }}
    .endpoint-column {{
      position: absolute;
      inset: 0 18px 0 auto;
      width: 190px;
      pointer-events: none;
    }}
    .endpoint-tag {{
      position: absolute;
      right: 18px;
      min-width: 160px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      padding: 8px 11px;
      border: 1px solid;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      backdrop-filter: blur(8px);
      opacity: 0;
      transform: translateX(16px);
      animation: slide-in 0.55s cubic-bezier(.2,.8,.2,1) forwards;
      animation-delay: 1.6s;
    }}
    .endpoint-name {{
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 112px;
    }}
    .chart-axis-title {{
      position: absolute;
      color: #746e64;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }}
    .chart-axis-title.x {{ left: 50%; bottom: 18px; transform: translateX(-50%); }}
    .chart-axis-title.y {{ left: 8px; top: 52%; transform: rotate(-90deg) translateY(50%); transform-origin: left top; }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    .rank-row, .time-row {{
      padding: 14px 0 0;
      border-top: 1px solid rgba(23, 23, 23, 0.08);
    }}
    .rank-row:first-child, .time-row:first-child {{
      border-top: 0;
      padding-top: 4px;
    }}
    .rank-head, .time-head {{
      display: flex;
      gap: 12px;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
    .rank-index {{
      font-family: var(--mono);
      font-size: 11px;
      color: #7b7468;
      letter-spacing: 0.12em;
    }}
    .rank-name, .time-name {{
      flex: 1;
      font-size: 15px;
      font-weight: 600;
    }}
    .rank-score, .time-score {{
      font-family: var(--mono);
      font-size: 13px;
    }}
    .rank-meta {{
      margin-top: 7px;
      color: #746e64;
      font-size: 12px;
    }}
    .rank-bar, .time-bar {{
      width: 100%;
      height: 8px;
      background: rgba(23, 23, 23, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }}
    .rank-bar span, .time-bar span {{
      display: block;
      height: 100%;
      width: 100%;
      transform-origin: left center;
      transform: scaleX(var(--fill));
      background: var(--rank-color);
      border-radius: inherit;
      animation: grow-bar 1s cubic-bezier(.2,.7,.2,1) both;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .table-wrap {{
      overflow: auto;
      border-radius: 20px;
      border: 1px solid rgba(23, 23, 23, 0.08);
      background: rgba(255,255,255,0.72);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 640px;
    }}
    th, td {{
      padding: 13px 14px;
      border-bottom: 1px solid rgba(23, 23, 23, 0.08);
      text-align: left;
      white-space: nowrap;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(252, 249, 243, 0.96);
      color: #6d685e;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 11px;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 10px;
      vertical-align: middle;
    }}
    .snapshot {{
      position: relative;
      border-radius: 22px;
      overflow: hidden;
      border: 1px solid rgba(23, 23, 23, 0.08);
      background: rgba(255,255,255,0.7);
    }}
    .snapshot img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .snapshot-badge {{
      position: absolute;
      left: 16px;
      top: 16px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.82);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(23,23,23,0.08);
      font-size: 12px;
      color: #5d584f;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    @keyframes draw-line {{
      to {{ stroke-dashoffset: 0; }}
    }}
    @keyframes grow-bar {{
      from {{ transform: scaleX(0); }}
      to {{ transform: scaleX(var(--fill)); }}
    }}
    @keyframes rise-in {{
      from {{ opacity: 0; transform: translateY(18px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fade-in {{
      to {{ opacity: 1; }}
    }}
    @keyframes pop-in {{
      from {{ opacity: 0; transform: scale(0.7); transform-origin: center; }}
      to {{ opacity: 1; transform: scale(1); }}
    }}
    @keyframes slide-in {{
      to {{ opacity: 1; transform: translateX(0); }}
    }}
    @media (max-width: 1120px) {{
      .hero,
      .dashboard,
      .metrics-grid {{
        grid-template-columns: 1fr;
      }}
      .hero-stats {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 780px) {{
      .page {{ width: min(100vw - 20px, 100%); padding-top: 18px; }}
      .hero-stats {{ grid-template-columns: 1fr; }}
      .panel {{ padding: 16px; border-radius: 20px; }}
      .chart-frame {{ min-height: 500px; }}
      .chart-svg {{ height: 500px; }}
      .endpoint-column {{ display: none; }}
      .table-wrap {{ margin-top: 12px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <div class="kicker">char lm optimizer report</div>
        <h1>TinyShakespeare<br>optimizer run</h1>
        <div class="subtitle">
          one long-form training comparison over the same model init and dataset split.
          the view below shows the full validation trajectory, not just end-state numbers.
        </div>
      </div>
      <div class="hero-stats">
        <div class="stat-card">
          <div class="stat-label">best validation</div>
          <div class="stat-value">{html.escape(leader['label'])}</div>
          <div class="stat-meta">{fmt_loss(leader['best_val'])} at step {fmt_step(leader['best_step'])}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">margin to next</div>
          <div class="stat-value">{fmt_loss(gap)}</div>
          <div class="stat-meta">{html.escape(runner_up['label']) if runner_up else "single run"}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">fastest wall time</div>
          <div class="stat-value">{html.escape(fastest['label'])}</div>
          <div class="stat-meta">{fmt_seconds(fastest['final_elapsed_s'])} to final checkpoint</div>
        </div>
      </div>
    </section>

    <section class="dashboard">
      <div class="panel">
        <div class="panel-title">
          <div>
            <h2>trajectory</h2>
            <div class="panel-strong-label">validation loss over time</div>
          </div>
          <div class="panel-note">{fmt_step(max_step)} steps · {len(summaries)} optimizers</div>
        </div>
        <div class="chart-frame">
          <div class="chart-caption">
            <div class="eyebrow">winner callout</div>
            <h3>{html.escape(leader['label'])} leads this saved run</h3>
            <p>
              best validation loss landed at {fmt_loss(leader['best_val'])}. final checkpoint finish belongs to
              {html.escape(sharpest_finish['label'])} at {fmt_loss(sharpest_finish['final_val'])}.
            </p>
          </div>
          <svg class="chart-svg" viewBox="0 0 {chart_w} {chart_h}" role="img" aria-label="validation loss chart">
            <rect x="0" y="0" width="{chart_w}" height="{chart_h}" fill="transparent"/>
            {''.join(grid_lines)}
            {''.join(x_ticks)}
            <line x1="{chart_left}" y1="{chart_bottom}" x2="{chart_right}" y2="{chart_bottom}" class="grid-line"/>
            <line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_bottom}" class="grid-line"/>
            {''.join(line_layers)}
          </svg>
          <div class="endpoint-column">
            {''.join(endpoint_labels)}
          </div>
          <div class="chart-axis-title x">training step</div>
          <div class="chart-axis-title y">validation loss</div>
        </div>
      </div>

      <div class="stack">
        <div class="panel">
          <div class="panel-title">
            <div>
              <h2>ranking</h2>
              <div class="panel-strong-label">best validation loss</div>
            </div>
            <div class="panel-note">lower is better</div>
          </div>
          {''.join(rank_cards)}
        </div>

        <div class="panel">
          <div class="panel-title">
            <div>
              <h2>runtime</h2>
              <div class="panel-strong-label">wall time to final checkpoint</div>
            </div>
            <div class="panel-note">same hardware path</div>
          </div>
          {''.join(time_cards)}
        </div>
      </div>
    </section>

    <section class="metrics-grid">
      <div class="panel">
        <div class="panel-title">
          <div>
            <h2>metrics table</h2>
            <div class="panel-strong-label">checkpoint summary</div>
          </div>
          <div class="panel-note">best / final / speed</div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>optimizer</th>
                <th>best val</th>
                <th>final val</th>
                <th>best step</th>
                <th>wall time</th>
                <th>tokens/sec</th>
              </tr>
            </thead>
            <tbody>
              {''.join(table_rows)}
            </tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">
          <div>
            <h2>static export</h2>
            <div class="panel-strong-label">png snapshot</div>
          </div>
          <div class="panel-note">shareable still</div>
        </div>
        <div class="snapshot">
          <div class="snapshot-badge">static artifact</div>
          <img src="{image_name}" alt="Static validation loss plot">
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(html_text, encoding="utf-8")


def render_gif_report(args, summaries):
    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    all_val_losses = [row["val_loss"] for summary in summaries for row in summary["rows"]]
    ymin = min(all_val_losses)
    ymax = max(all_val_losses)
    yrange = max(ymax - ymin, 1e-6)
    ymin -= yrange * 0.05
    ymax += yrange * 0.08

    fig = plt.figure(figsize=(12.8, 6.6), facecolor="#0b1020")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3.6, 1.35], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])

    ax.set_facecolor("#121a2b")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", linewidth=0.9, alpha=0.55)
    ax.grid(axis="x", color="#1e293b", linewidth=0.7, alpha=0.55)
    ax.set_xlabel("training step", fontsize=11, color="#cbd5e1")
    ax.set_ylabel("validation loss", fontsize=11, color="#cbd5e1")
    ax.tick_params(colors="#94a3b8", labelsize=10)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_xlim(0, max_step)
    ax.set_ylim(ymin, ymax)

    ax_side.set_facecolor("#111827")
    for spine in ax_side.spines.values():
        spine.set_color("#374151")
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)

    leader = summaries[0]
    ax.set_title("TinyShakespeare validation loss", loc="left", fontsize=18, fontweight="bold", color="#f8fafc", pad=16)
    subtitle = ax.text(
        0.0,
        1.005,
        f"animated training trajectory · leader: {leader['label']} at {leader['best_val']:.3f}",
        transform=ax.transAxes,
        fontsize=9.6,
        color="#94a3b8",
        va="bottom",
    )
    ax.text(
        0.995,
        1.005,
        f"{fmt_step(max_step)} steps",
        transform=ax.transAxes,
        fontsize=9.6,
        color="#94a3b8",
        va="bottom",
        ha="right",
    )

    ax_side.text(0.08, 0.91, leader["label"], fontsize=24, color="#f8fafc", fontweight="bold")
    side_best = ax_side.text(
        0.08,
        0.83,
        f"best val {leader['best_val']:.3f}\nat step {fmt_step(leader['best_step'])}",
        fontsize=10.2,
        color="#cbd5e1",
        linespacing=1.35,
    )
    progress_text = ax_side.text(
        0.08,
        0.745,
        "progress 0%",
        fontsize=10,
        color="#94a3b8",
        family="monospace",
    )
    progress_bar_bg = plt.Rectangle((0.08, 0.712), 0.84, 0.018, facecolor="#1f2937", edgecolor="none")
    progress_bar = plt.Rectangle((0.08, 0.712), 0.0, 0.018, facecolor="#e2e8f0", edgecolor="none")
    ax_side.add_patch(progress_bar_bg)
    ax_side.add_patch(progress_bar)

    line_artists = {}
    point_artists = {}
    panel_value_artists = {}
    panel_delta_artists = {}
    row_y_positions = [0.61 - i * 0.086 for i in range(len(summaries))]

    for idx, summary in enumerate(summaries):
        color = PALETTE.get(summary["optimizer"], "#334155")
        (line,) = ax.plot([], [], color=color, linewidth=1.8, solid_capstyle="round")
        point = ax.scatter([], [], s=24, color=color, edgecolors="#e2e8f0", linewidths=0.9, zorder=5)
        line_artists[summary["optimizer"]] = line
        point_artists[summary["optimizer"]] = point

        y = row_y_positions[idx]
        ax_side.add_patch(plt.Rectangle((0.08, y - 0.012), 0.022, 0.022, facecolor=color, edgecolor="none"))
        ax_side.text(0.12, y, summary["label"], va="center", fontsize=10.2, color="#e5e7eb", fontweight="semibold")
        panel_value_artists[summary["optimizer"]] = ax_side.text(
            0.92,
            y,
            "....",
            ha="right",
            va="center",
            fontsize=10.8,
            color=color,
            family="monospace",
            fontweight="semibold",
        )
        panel_delta_artists[summary["optimizer"]] = ax_side.text(
            0.92,
            y - 0.032,
            "",
            ha="right",
            va="center",
            fontsize=8.0,
            color="#94a3b8",
            family="monospace",
        )

    total_points = max(len(summary["rows"]) for summary in summaries)
    hold_frames = 20
    total_frames = total_points + hold_frames

    def update(frame_idx):
        active_points = min(frame_idx + 1, total_points)
        progress = active_points / max(total_points, 1)

        for summary in summaries:
            rows = summary["rows"][:active_points]
            xs = [row["step"] for row in rows]
            ys = [row["val_loss"] for row in rows]
            line_artists[summary["optimizer"]].set_data(xs, ys)
            if rows:
                point_artists[summary["optimizer"]].set_offsets([[xs[-1], ys[-1]]])
                panel_value_artists[summary["optimizer"]].set_text(f"{ys[-1]:.3f}")
                panel_delta_artists[summary["optimizer"]].set_text(f"best {summary['best_val']:.3f}")
            else:
                point_artists[summary["optimizer"]].set_offsets([])
                panel_value_artists[summary["optimizer"]].set_text("....")
                panel_delta_artists[summary["optimizer"]].set_text("")

        progress_bar.set_width(0.84 * progress)
        progress_text.set_text(f"progress {progress * 100:5.1f}%")

        if frame_idx >= total_points - 1:
            subtitle.set_text(
                f"animated training trajectory · winner: {leader['label']} · best val {leader['best_val']:.3f}"
            )
            progress_text.set_text("progress 100.0% · final frame")
            side_best.set_text(
                f"best val {leader['best_val']:.3f}\nat step {fmt_step(leader['best_step'])}"
            )

        artists = (
            list(line_artists.values())
            + list(point_artists.values())
            + list(panel_value_artists.values())
            + list(panel_delta_artists.values())
            + [subtitle, side_best, progress_text, progress_bar]
        )
        return artists

    anim = FuncAnimation(fig, update, frames=total_frames, interval=170, blit=False)
    args.gif_out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.90, left=0.08, right=0.97, bottom=0.12)
    anim.save(args.gif_out, writer=PillowWriter(fps=8))
    plt.close(fig)


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
    plt.close(fig)
    render_html_report(args, summaries)
    render_gif_report(args, summaries)
    print(f"wrote {args.out}")
    print(f"wrote {args.html_out}")
    print(f"wrote {args.gif_out}")


if __name__ == "__main__":
    main()
