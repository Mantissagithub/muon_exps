import argparse
import csv
import html
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/muon_exps_matplotlib")

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "char_lm" / "results.csv"

DISPLAY_NAMES = {
    "adamw": "AdamW",
    "torch_muon": "Torch Muon",
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
    "adamw": "#1f77b4",
    "torch_muon": "#ff7f0e",
    "muon_like": "#8c564b",
    "normuon": "#9467bd",
    "u_normuon": "#d62728",
    "aurora": "#7f7f7f",
    "riemann_aurora": "#17becf",
    "adafactor": "#bcbd22",
    "sf_adamw": "#2ca02c",
    "amuse_muon": "#e41a1c",
    "sf_muon_fixed_beta_0.6": "#4daf4a",
    "sf_muon_fixed_beta_0.9": "#984ea3",
    "lion": "#ca8a04",
    "adopt": "#b91c1c",
    "prodigy": "#0891b2",
    "soap": "#111827",
    "mars": "#a855f7",
    "sophia": "#4f46e5",
    "ademamix": "#9333ea",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--html-out", type=Path)
    parser.add_argument("--beta-out", type=Path)
    parser.add_argument("--distance-out", type=Path)
    parser.add_argument("--cosine-out", type=Path)
    parser.add_argument("--val-components-out", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.results.parent
    args.out = args.out if args.out is not None else out_dir / "loss_curves.png"
    args.html_out = args.html_out if args.html_out is not None else out_dir / "loss_report.html"
    args.beta_out = args.beta_out if args.beta_out is not None else out_dir / "beta_schedule.png"
    args.distance_out = args.distance_out if args.distance_out is not None else out_dir / "amuse_distances.png"
    args.cosine_out = args.cosine_out if args.cosine_out is not None else out_dir / "update_cosine.png"
    args.val_components_out = (
        args.val_components_out if args.val_components_out is not None else out_dir / "amuse_val_components.png"
    )
    return args


def parse_float(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def load_results(path: Path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    by_opt = {}
    for row in rows:
        parsed = {
            "optimizer": row["optimizer"],
            "beta1": parse_float(row, "beta1"),
            "rho": parse_float(row, "rho"),
            "fixed_beta": parse_float(row, "fixed_beta"),
            "step": int(row["step"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "val_loss_x": parse_float(row, "val_loss_x"),
            "val_loss_z": parse_float(row, "val_loss_z"),
            "val_loss_y": parse_float(row, "val_loss_y"),
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
            "y_x_expected_distance": parse_float(row, "y_x_expected_distance"),
            "y_z_expected_distance": parse_float(row, "y_z_expected_distance"),
            "amuse_y_reconstruction_error": parse_float(row, "amuse_y_reconstruction_error"),
        }
        by_opt.setdefault(parsed["optimizer"], []).append(parsed)

    for opt_rows in by_opt.values():
        opt_rows.sort(key=lambda row: row["step"])
    return by_opt


def summarize(by_opt):
    summaries = []
    for optimizer, rows in by_opt.items():
        best_idx, best_row = min(enumerate(rows), key=lambda item: item[1]["val_loss"])
        final_row = rows[-1]
        summaries.append(
            {
                "optimizer": optimizer,
                "label": DISPLAY_NAMES.get(optimizer, optimizer),
                "beta1": final_row["beta1"],
                "rho": final_row["rho"],
                "fixed_beta": final_row["fixed_beta"],
                "rows": rows,
                "best_idx": best_idx,
                "best_step": best_row["step"],
                "best_val": best_row["val_loss"],
                "final_val": final_row["val_loss"],
                "final_elapsed_s": final_row["elapsed_s"],
            }
        )
    summaries.sort(key=lambda item: item["best_val"])
    return summaries


def has_metric(summaries, key):
    return any(math.isfinite(row.get(key, float("nan"))) for summary in summaries for row in summary["rows"])


def fmt_loss(value):
    return f"{value:.3f}"


def fmt_step(value):
    return f"{value:,}"


def fmt_seconds(value):
    if value >= 60:
        minutes = int(value // 60)
        seconds = int(round(value - minutes * 60))
        return f"{minutes}m {seconds:02d}s"
    return f"{value:.1f}s"


def style_axis(ax, ylabel):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.8)
    ax.grid(True, color="#b0b0b0", linewidth=0.6, alpha=0.65)
    ax.tick_params(colors="#222222", labelsize=8, width=0.6, length=3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Step", fontsize=9, color="#222222")
    ax.set_ylabel(ylabel, fontsize=9, color="#222222")


def focus_subset(summaries):
    available = {summary["optimizer"]: summary for summary in summaries}
    preferred = [
        "adamw",
        "torch_muon",
        "muon_like",
        "sf_muon_fixed_beta_0.6",
        "sf_muon_fixed_beta_0.9",
        "amuse_muon_b0.4_r0.5",
        "amuse_muon_b0.4_r0.8",
        "amuse_muon_b0.6_r0.5",
        "amuse_muon_b0.6_r0.8",
        "amuse_muon",
    ]
    focus = [available[name] for name in preferred if name in available]
    if focus:
        return focus
    return summaries[: min(4, len(summaries))]


def render_main_png(args, summaries):
    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    losses = [row["val_loss"] for summary in summaries for row in summary["rows"]]
    ymin = min(losses)
    ymax = max(losses)
    span = max(ymax - ymin, 1e-6)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"]})
    fig = plt.figure(figsize=(8.6, 4.8), dpi=220, facecolor="white")
    ax = fig.add_subplot(1, 1, 1)

    style_axis(ax, "Validation Loss")
    ax.set_title("TinyShakespeare (char LM)", fontsize=12, pad=7)
    ax.set_xlim(0, max_step)
    ax.set_ylim(max(0.0, ymin - span * 0.04), ymax + span * 0.03)

    ordered = sorted(summaries, key=lambda item: item["best_val"])
    for idx, summary in enumerate(ordered):
        color = PALETTE.get(summary["optimizer"], "#334155")
        steps = [row["step"] for row in summary["rows"]]
        vals = [row["val_loss"] for row in summary["rows"]]
        ax.plot(
            steps,
            vals,
            color=color,
            linewidth=1.45 if idx == 0 else 1.15,
            alpha=0.98 if idx == 0 else 0.92,
            label=summary["label"],
        )

    zoom_start = int(max_step * 0.72)
    inset = ax.inset_axes([0.59, 0.17, 0.36, 0.32])
    style_axis(inset, "")
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.tick_params(labelsize=6)
    inset.set_xlim(zoom_start, max_step)

    zoom_vals = []
    for idx, summary in enumerate(ordered):
        color = PALETTE.get(summary["optimizer"], "#334155")
        rows = [row for row in summary["rows"] if row["step"] >= zoom_start]
        steps = [row["step"] for row in rows]
        vals = [row["val_loss"] for row in rows]
        zoom_vals.extend(vals)
        inset.plot(
            steps,
            vals,
            color=color,
            linewidth=1.2 if idx == 0 else 0.95,
            alpha=0.96,
        )

    if zoom_vals:
        zmin = min(zoom_vals)
        zmax = max(zoom_vals)
        zspan = max(zmax - zmin, 1e-6)
        inset.set_ylim(zmin - zspan * 0.08, zmax + zspan * 0.08)

    legend = ax.legend(loc="upper right", frameon=True, fontsize=7, ncol=2)
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def render_beta_png(args, summaries):
    if not has_metric(summaries, "beta_t"):
        return False

    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"]})
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220, facecolor="white")
    style_axis(ax, "beta_t")
    ax.set_xlim(0, max_step)
    ax.set_ylim(0.0, 1.02)

    for summary in summaries:
        rows = [row for row in summary["rows"] if math.isfinite(row.get("beta_t", float("nan")))]
        if not rows:
            continue
        xs = [row["step"] for row in rows]
        ys = [row["beta_t"] for row in rows]
        linestyle = "--" if math.isfinite(summary["fixed_beta"]) else "-"
        ax.plot(xs, ys, color=PALETTE.get(summary["optimizer"], "#334155"), linewidth=1.25, linestyle=linestyle, label=summary["label"])

    legend = ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=2.2)
    for text in legend.get_texts():
        text.set_color("#222222")
    fig.text(0.09, 0.95, "AMUSE beta schedule", color="#111111", fontsize=9.8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.22)
    args.beta_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.beta_out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return True


def render_cosine_png(args, summaries):
    if not has_metric(summaries, "update_cosine_similarity"):
        return False

    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"]})
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220, facecolor="white")
    style_axis(ax, "cosine similarity")
    ax.set_xlim(0, max_step)
    ax.set_ylim(-1.0, 1.0)

    cosine_values = []
    for summary in summaries:
        rows = [row for row in summary["rows"] if math.isfinite(row.get("update_cosine_similarity", float("nan")))]
        if not rows:
            continue
        xs = [row["step"] for row in rows]
        ys = [max(-1.0, min(1.0, row["update_cosine_similarity"])) for row in rows]
        cosine_values.extend(ys)
        ax.plot(
            xs,
            ys,
            color=PALETTE.get(summary["optimizer"], "#334155"),
            linewidth=1.35,
            marker="o",
            markersize=2.6,
            markeredgewidth=0.0,
            label=summary["label"],
        )

    if cosine_values:
        cmin = min(cosine_values)
        cmax = max(cosine_values)
        if cmin > 0.95:
            ax.set_ylim(0.95, 1.001)
        else:
            pad = max((cmax - cmin) * 0.08, 0.01)
            ax.set_ylim(max(-1.0, cmin - pad), min(1.0, cmax + pad))

    legend = ax.legend(loc="best", frameon=False, fontsize=7, handlelength=2.2)
    for text in legend.get_texts():
        text.set_color("#222222")
    fig.text(0.09, 0.95, "x-update cosine similarity", color="#111111", fontsize=9.8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.22)
    args.cosine_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.cosine_out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return True


def render_distance_png(args, summaries):
    keys = [
        ("z_x_distance", "||z - x|| / ||x||"),
        ("y_x_distance", "||y - x|| / ||x||"),
        ("y_z_distance", "||y - z|| / ||z||"),
    ]
    if not any(has_metric(summaries, key) for key, _ in keys):
        return False

    max_step = max(summary["rows"][-1]["step"] for summary in summaries)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"]})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), dpi=220, facecolor="white", sharex=True)

    for ax, (key, label) in zip(axes, keys):
        style_axis(ax, label)
        ax.set_xlim(0, max_step)
        for summary in summaries:
            rows = [row for row in summary["rows"] if math.isfinite(row.get(key, float("nan")))]
            if not rows:
                continue
            xs = [row["step"] for row in rows]
            ys = [row[key] for row in rows]
            ax.plot(xs, ys, color=PALETTE.get(summary["optimizer"], "#334155"), linewidth=1.15, label=summary["label"])
            expected_key = {
                "y_x_distance": "y_x_expected_distance",
                "y_z_distance": "y_z_expected_distance",
            }.get(key)
            if expected_key is not None:
                exp_rows = [row for row in rows if math.isfinite(row.get(expected_key, float("nan")))]
                if exp_rows:
                    ax.plot(
                        [row["step"] for row in exp_rows],
                        [row[expected_key] for row in exp_rows],
                        color=PALETTE.get(summary["optimizer"], "#334155"),
                        linewidth=0.9,
                        linestyle="--",
                        alpha=0.7,
                    )
        ax.legend(loc="best", frameon=False, fontsize=6, handlelength=2.0)

    axes[-1].set_xlabel("Step", fontsize=9, color="#222222")
    fig.text(0.09, 0.97, "AMUSE sequence distances", color="#111111", fontsize=9.8)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.08, hspace=0.28)
    args.distance_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.distance_out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return True


def render_val_components_png(args, summaries):
    component_summaries = [
        summary for summary in summaries if any(math.isfinite(row.get("val_loss_y", float("nan"))) for row in summary["rows"])
    ]
    if not component_summaries:
        return False

    max_step = max(summary["rows"][-1]["step"] for summary in component_summaries)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"]})
    fig, axes = plt.subplots(len(component_summaries), 1, figsize=(7.2, 2.4 * len(component_summaries)), dpi=220, facecolor="white", sharex=True)
    if len(component_summaries) == 1:
        axes = [axes]

    components = [
        ("val_loss_x", "x_t", "-"),
        ("val_loss_y", "y_t", "--"),
        ("val_loss_z", "z_t", ":"),
    ]
    for ax, summary in zip(axes, component_summaries):
        style_axis(ax, "validation loss")
        ax.set_xlim(0, max_step)
        for key, label, linestyle in components:
            rows = [row for row in summary["rows"] if math.isfinite(row.get(key, float("nan")))]
            if not rows:
                continue
            ax.plot(
                [row["step"] for row in rows],
                [row[key] for row in rows],
                color=PALETTE.get(summary["optimizer"], "#334155"),
                linewidth=1.15,
                linestyle=linestyle,
                label=label,
            )
        ax.set_title(summary["label"], fontsize=9, pad=5)
        ax.legend(loc="best", frameon=False, fontsize=7, handlelength=2.2)

    axes[-1].set_xlabel("Step", fontsize=9, color="#222222")
    fig.text(0.09, 0.97, "AMUSE validation components", color="#111111", fontsize=9.8)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.10, hspace=0.35)
    args.val_components_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.val_components_out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return True


def render_html_report(args, summaries):
    leader = summaries[0]
    runner_up = summaries[1] if len(summaries) > 1 else None
    image_name = html.escape(args.out.name)
    beta_name = html.escape(args.beta_out.name)
    distance_name = html.escape(args.distance_out.name)
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
      Results rendered directly from <code>artifacts/char_lm/results.csv</code>.
      The main figure compares validation loss across the optimizers present in that file.
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
        The left panel shows every optimizer present in the CSV; the right panel focuses on the main comparison set when available.
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
        <div class="caption"><strong>Figure 2.</strong> AMUSE beta schedule.</div>
      </div>
      <div class="figure">
        <img src="{distance_name}" alt="AMUSE sequence distances">
        <div class="caption"><strong>Figure 3.</strong> Relative x/y/z sequence distances.</div>
      </div>
      <div class="figure">
        <img src="{cosine_name}" alt="AMUSE update cosine similarity">
        <div class="caption"><strong>Figure 4.</strong> Cosine similarity between successive AMUSE x-updates.</div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(html_text, encoding="utf-8")


def main():
    args = parse_args()
    by_opt = load_results(args.results)
    if not by_opt:
        raise ValueError(f"no rows found in {args.results}")

    summaries = summarize(by_opt)
    render_main_png(args, summaries)
    render_html_report(args, summaries)
    wrote_beta = render_beta_png(args, summaries)
    wrote_distance = render_distance_png(args, summaries)
    wrote_cosine = render_cosine_png(args, summaries)
    wrote_val_components = render_val_components_png(args, summaries)
    print(f"wrote {args.out}")
    print(f"wrote {args.html_out}")
    if wrote_beta:
        print(f"wrote {args.beta_out}")
    if wrote_distance:
        print(f"wrote {args.distance_out}")
    if wrote_cosine:
        print(f"wrote {args.cosine_out}")
    if wrote_val_components:
        print(f"wrote {args.val_components_out}")


if __name__ == "__main__":
    main()
