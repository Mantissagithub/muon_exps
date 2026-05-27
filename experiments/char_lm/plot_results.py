import argparse
import csv
import html
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
      --bg: #f6f4ee;
      --panel: rgba(255, 255, 255, 0.84);
      --panel-strong: #ffffff;
      --ink: #171717;
      --muted: #667085;
      --hairline: rgba(23, 23, 23, 0.12);
      --shadow: 0 24px 70px rgba(22, 30, 45, 0.10);
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
        radial-gradient(circle at 16% 0%, rgba(255, 107, 0, 0.10), transparent 30%),
        radial-gradient(circle at 84% 8%, rgba(0, 194, 168, 0.09), transparent 28%),
        linear-gradient(180deg, #fbfaf7 0%, var(--bg) 100%);
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
      grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.78fr);
      gap: 18px;
    }}
    .panel {{
      padding: 22px;
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
      min-height: 0;
      border-radius: 22px;
      background: #ffffff;
      border: 1px solid rgba(23, 23, 23, 0.08);
      overflow: hidden;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.80);
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
      background: rgba(255,255,255,0.82);
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
    .read-card {{
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .read-pill {{
      border: 1px solid rgba(23,23,23,0.08);
      border-radius: 18px;
      background: rgba(255,255,255,0.72);
      padding: 14px;
    }}
    .read-label {{
      color: #7a7468;
      text-transform: uppercase;
      letter-spacing: 0.13em;
      font-size: 10px;
    }}
    .read-value {{
      margin-top: 8px;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .read-note {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
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
      .read-card {{ grid-template-columns: 1fr; }}
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
            <h2>static trajectory</h2>
            <div class="panel-strong-label">validation loss with late-run zoom</div>
          </div>
          <div class="panel-note">{fmt_step(max_step)} steps · {len(summaries)} optimizers</div>
        </div>
        <div class="chart-frame">
          <img src="{image_name}" alt="Static validation loss plot" style="display:block;width:100%;height:auto;">
        </div>
        <div class="read-card">
          <div class="read-pill">
            <div class="read-label">winner</div>
            <div class="read-value">{html.escape(leader['label'])} · {fmt_loss(leader['best_val'])}</div>
            <div class="read-note">best validation at step {fmt_step(leader['best_step'])}</div>
          </div>
          <div class="read-pill">
            <div class="read-label">next best gap</div>
            <div class="read-value">{fmt_loss(gap)}</div>
            <div class="read-note">margin over {html.escape(runner_up['label']) if runner_up else "the next run"}</div>
          </div>
          <div class="read-pill">
            <div class="read-label">best final checkpoint</div>
            <div class="read-value">{html.escape(sharpest_finish['label'])}</div>
            <div class="read-note">final val {fmt_loss(sharpest_finish['final_val'])}</div>
          </div>
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
            <h2>artifact note</h2>
            <div class="panel-strong-label">what to share</div>
          </div>
          <div class="panel-note">static only</div>
        </div>
        <div class="read-card" style="grid-template-columns:1fr;">
          <div class="read-pill">
            <div class="read-label">png</div>
            <div class="read-value">loss_curves.png</div>
            <div class="read-note">clean static chart, good for README and quick sharing.</div>
          </div>
          <div class="read-pill">
            <div class="read-label">html</div>
            <div class="read-value">loss_report.html</div>
            <div class="read-note">same numbers with ranking, runtime, and checkpoint table.</div>
          </div>
        </div>
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

    fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=190, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#e5e7eb")
        spine.set_linewidth(1.0)

    ax.grid(axis="y", color="#edf1f5", linewidth=0.75)
    ax.grid(axis="x", color="#f5f7fa", linewidth=0.55)
    ax.tick_params(colors="#a3acba", labelsize=8)
    ax.set_xlabel("Training step", color="#8b95a7", fontsize=8, labelpad=10)
    ax.set_ylabel("Loss", color="#8b95a7", fontsize=8, labelpad=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0k"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xlim(0, max_step)
    ax.set_ylim(max(0.0, ymin - span * 0.05), ymax + span * 0.04)

    # Keep the main view readable, then use the inset for the late-run differences.
    show_order = sorted(summaries, key=lambda item: item["best_val"])
    for summary in show_order:
        color = PALETTE.get(summary["optimizer"], "#334155")
        alpha = 0.98 if summary is summaries[0] else 0.72
        width = 1.85 if summary is summaries[0] else 1.25
        steps = [row["step"] for row in summary["rows"]]
        vals = [row["val_loss"] for row in summary["rows"]]
        ax.plot(steps, vals, color=color, linewidth=width, alpha=alpha, label=summary["label"], solid_capstyle="round")

    zoom_start = int(max_step * 0.62)
    inset = ax.inset_axes([0.64, 0.43, 0.32, 0.34])
    inset.set_facecolor("#ffffff")
    for spine in inset.spines.values():
        spine.set_color("#e5e7eb")
        spine.set_linewidth(0.8)
    inset.grid(axis="y", color="#edf1f5", linewidth=0.55)
    inset.grid(axis="x", color="#f5f7fa", linewidth=0.45)
    inset.tick_params(colors="#a3acba", labelsize=6, length=2)
    inset.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
    inset.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    inset.set_xlim(zoom_start, max_step)

    zoom_vals = []
    for summary in show_order:
        rows = [row for row in summary["rows"] if row["step"] >= zoom_start]
        steps = [row["step"] for row in rows]
        vals = [row["val_loss"] for row in rows]
        zoom_vals.extend(vals)
        color = PALETTE.get(summary["optimizer"], "#334155")
        width = 1.5 if summary is summaries[0] else 1.0
        inset.plot(steps, vals, color=color, linewidth=width, alpha=0.92, solid_capstyle="round")

    if zoom_vals:
        zmin = min(zoom_vals)
        zmax = max(zoom_vals)
        zspan = max(zmax - zmin, 1e-6)
        inset.set_ylim(zmin - zspan * 0.12, zmax + zspan * 0.18)
    ax.indicate_inset_zoom(inset, edgecolor="#e5e7eb", linewidth=0.9, alpha=0.9)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(4, len(show_order)),
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color("#4b5563")

    fig.text(0.075, 0.935, "TinyShakespeare validation loss", color="#111827", fontsize=16, fontweight="bold")
    fig.text(
        0.075,
        0.905,
        f"{fmt_step(max_step)} steps · {len(summaries)} optimizers · inset shows the late-run spread",
        color="#8b95a7",
        fontsize=8.5,
    )
    fig.text(
        0.73,
        0.905,
        f"best: {summaries[0]['label']} {summaries[0]['best_val']:.3f}",
        color=PALETTE.get(summaries[0]["optimizer"], "#111827"),
        fontsize=9,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.20)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    args = parse_args()
    by_opt = load_results(args.results)
    if not by_opt:
        raise ValueError(f"no rows found in {args.results}")

    summaries = summarize(by_opt)
    render_clean_png(args, summaries)
    render_html_report(args, summaries)
    print(f"wrote {args.out}")
    print(f"wrote {args.html_out}")


if __name__ == "__main__":
    main()
