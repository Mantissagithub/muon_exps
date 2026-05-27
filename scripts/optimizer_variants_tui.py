# run via: uv run scripts/optimizer_variants_tui.py
# rich tui wrapper for cuda/benchmark_optimizer_variants.cu.

import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "cuda" / "benchmark_optimizer_variants.cu"
BIN = ROOT / "artifacts" / "bin" / "benchmark_optimizer_variants"
CUDA_DEPS = [
  SRC,
  ROOT / "cuda" / "muon.cu",
  ROOT / "cuda" / "normuon.cu",
  ROOT / "cuda" / "u_normuon.cu",
  ROOT / "cuda" / "aurora.cu",
  ROOT / "cuda" / "riemann_aurora.cu",
]

WHITE = "white"
LIGHT = "grey85"
MID = "grey70"
DARK = "grey50"
DIM = "grey35"


def ensure_binary(console: Console) -> None:
  if BIN.exists() and BIN.stat().st_mtime >= max(path.stat().st_mtime for path in CUDA_DEPS):
    return
  console.print(f"[{MID}]optimizer benchmark binary not found or stale, compiling...[/]")
  if shutil.which("nvcc") is None:
    console.print(f"[{WHITE}]nvcc not on PATH - install CUDA toolkit[/]")
    sys.exit(1)
  BIN.parent.mkdir(parents=True, exist_ok=True)
  rc = subprocess.call(
    [
      "nvcc",
      "-O3",
      "-std=c++17",
      str(SRC),
      "-lcublas",
      "-o",
      str(BIN),
    ],
    cwd=str(ROOT),
  )
  if rc != 0 or not BIN.exists():
    console.print(f"[{WHITE}]compile failed[/]")
    sys.exit(1)


class State:
  def __init__(self) -> None:
    self.rows: list[dict[str, str]] = []
    self.current_optimizer: str | None = None
    self.current_shape: str | None = None
    self.t0 = time.monotonic()
    self.done = False
    self.unknown_lines: list[str] = []

  def elapsed_s(self) -> float:
    return time.monotonic() - self.t0


def cell_running() -> Spinner:
  return Spinner("dots", text=Text("running", style=DIM), style=DIM)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
  try:
    return float(row[key])
  except (KeyError, ValueError):
    return default


def fmt_num(row: dict[str, str], key: str, digits: int = 3) -> str:
  try:
    return f"{float(row[key]):.{digits}f}"
  except (KeyError, ValueError):
    return "-"


def shape_key(row: dict[str, str]) -> str:
  return f"{row['N']}x{row['M']}"


def speed_cell(row: dict[str, str], baseline_ms: float | None) -> Text:
  ms = as_float(row, "avg_step_ms")
  label = f"{ms:.2f} ms"
  if baseline_ms and row["optimizer"] != "muon":
    label += f"  {ms / baseline_ms:.1f}x"
  elif row["optimizer"] == "muon":
    label += "  base"
  return Text(label, style=LIGHT)


def row_balance_cell(row: dict[str, str]) -> Text:
  cv = as_float(row, "row_norm_cv")
  if cv < 0.10:
    label = "flat"
    style = f"bold {WHITE}"
  elif cv < 0.60:
    label = "mixed"
    style = LIGHT
  else:
    label = "uneven"
    style = MID
  return Text(f"{label}  cv {cv:.3f}", style=style)


def dead_rows_cell(row: dict[str, str]) -> Text:
  dead = as_float(row, "dead_row_fraction")
  if dead == 0.0:
    return Text("none", style=f"bold {WHITE}")
  return Text(f"{dead * 100:.1f}%", style=MID)


def geometry_cell(row: dict[str, str]) -> Text:
  defect = as_float(row, "orthogonality_defect")
  if defect < 0.50:
    label = "kept"
    style = f"bold {WHITE}"
  elif defect < 1.50:
    label = "drift"
    style = LIGHT
  else:
    label = "bad"
    style = MID
  return Text(f"{label}  {defect:.3f}", style=style)


def direction_cell(row: dict[str, str]) -> Text:
  align = as_float(row, "gradient_alignment")
  if align >= 0.85:
    label = "tracks"
    style = f"bold {WHITE}"
  elif align >= 0.50:
    label = "loose"
    style = LIGHT
  else:
    label = "weak"
    style = MID
  return Text(f"{label}  {align:.3f}", style=style)


def take_cell(row: dict[str, str], baseline_ms: float | None) -> Text:
  opt = row["optimizer"]
  cv = as_float(row, "row_norm_cv")
  align = as_float(row, "gradient_alignment")
  ms = as_float(row, "avg_step_ms")
  if opt == "muon":
    return Text("fast baseline", style=LIGHT)
  if opt in {"normuon", "u_normuon"}:
    return Text("same shape as muon", style=MID)
  if opt == "aurora":
    return Text("kills dead rows, loses direction", style=MID if align < 0.5 else LIGHT)
  if opt == "riemann_aurora":
    if baseline_ms and ms > baseline_ms * 20:
      return Text("balanced but very slow", style=f"bold {WHITE}" if cv < 0.1 else LIGHT)
    return Text("balanced rows", style=f"bold {WHITE}" if cv < 0.1 else LIGHT)
  return Text("-", style=DIM)


def wins_by_shape(rows: list[dict[str, str]]) -> dict[str, dict[str, set[str]]]:
  out: dict[str, dict[str, set[str]]] = {}
  shapes = sorted({shape_key(row) for row in rows})
  metrics = {
    "speed": ("avg_step_ms", min),
    "balance": ("row_norm_cv", min),
    "dead": ("dead_row_fraction", min),
    "geometry": ("orthogonality_defect", min),
    "direction": ("gradient_alignment", max),
  }
  for shape in shapes:
    shape_rows = [row for row in rows if shape_key(row) == shape]
    out[shape] = {name: set() for name in metrics}
    for name, (key, chooser) in metrics.items():
      vals = [as_float(row, key) for row in shape_rows]
      best = chooser(vals)
      tol = max(abs(best) * 1e-5, 1e-6)
      for row in shape_rows:
        if abs(as_float(row, key) - best) <= tol:
          out[shape][name].add(row["optimizer"])
  return out


def win_badges(row: dict[str, str], wins: dict[str, dict[str, set[str]]]) -> Text:
  shape_wins = wins.get(shape_key(row), {})
  labels = []
  for metric, label in [
    ("speed", "speed"),
    ("balance", "rows"),
    ("dead", "dead"),
    ("geometry", "geom"),
    ("direction", "dir"),
  ]:
    if row["optimizer"] in shape_wins.get(metric, set()):
      labels.append(label)
  if not labels:
    return Text("-", style=DIM)
  return Text("wins " + ",".join(labels), style=f"bold {WHITE}")


def baseline_by_shape(rows: list[dict[str, str]]) -> dict[str, float]:
  out: dict[str, float] = {}
  for row in rows:
    if row["optimizer"] == "muon":
      out[shape_key(row)] = as_float(row, "avg_step_ms")
  return out


def build_table(s: State) -> Table:
  t = Table(
    title=f"[bold {WHITE}]OPTIMIZER VARIANTS[/]  [{MID}]synthetic anisotropic update benchmark  ·  lower row cv/dead/ortho is better, higher align is better[/]",
    border_style=DARK,
    box=box.ROUNDED,
    expand=False,
    show_lines=True,
    pad_edge=False,
    title_justify="left",
  )
  t.add_column("shape", style=MID, no_wrap=True)
  t.add_column("optimizer", style=LIGHT, no_wrap=True)
  t.add_column("speed", justify="right", width=16)
  t.add_column("row balance", justify="right", width=16)
  t.add_column("dead rows", justify="right", width=10)
  t.add_column("geometry", justify="right", width=14)
  t.add_column("direction", justify="right", width=14)
  t.add_column("wins", style=LIGHT, no_wrap=True)
  t.add_column("read", style=MID, no_wrap=True)

  baselines = baseline_by_shape(s.rows)
  wins = wins_by_shape(s.rows)
  for row in s.rows:
    shape = shape_key(row)
    baseline_ms = baselines.get(shape)
    t.add_row(
      shape,
      row["optimizer"],
      speed_cell(row, baseline_ms),
      row_balance_cell(row),
      dead_rows_cell(row),
      geometry_cell(row),
      direction_cell(row),
      win_badges(row, wins),
      take_cell(row, baseline_ms),
    )

  if not s.done and s.current_optimizer and s.current_shape:
    t.add_row(
      Text(s.current_shape, style=DIM),
      Text(s.current_optimizer, style=DIM),
      cell_running(),
      Text("-", style=DIM),
      Text("-", style=DIM),
      Text("-", style=DIM),
      Text("-", style=DIM),
      Text("-", style=DIM),
      Text("-", style=DIM),
    )

  return t


def build_summary(s: State) -> Table:
  t = Table(
    title=f"[bold {WHITE}]READOUT[/]",
    border_style=DARK,
    box=box.SIMPLE,
    expand=False,
    show_header=True,
    pad_edge=False,
    title_justify="left",
  )
  t.add_column("shape", style=MID, no_wrap=True)
  t.add_column("speed winner", style=LIGHT, no_wrap=True)
  t.add_column("row winner", style=LIGHT, no_wrap=True)
  t.add_column("geometry winner", style=LIGHT, no_wrap=True)
  t.add_column("direction winner", style=LIGHT, no_wrap=True)

  seen: list[str] = []
  for row in s.rows:
    key = shape_key(row)
    if key not in seen:
      seen.append(key)

  for key in seen:
    rows = [row for row in s.rows if shape_key(row) == key]
    fastest = min(rows, key=lambda r: as_float(r, "avg_step_ms"))
    flattest = min(rows, key=lambda r: as_float(r, "row_norm_cv"))
    geometry = min(rows, key=lambda r: as_float(r, "orthogonality_defect"))
    aligned = max(rows, key=lambda r: as_float(r, "gradient_alignment"))
    t.add_row(
      key,
      f"{fastest['optimizer']} ({as_float(fastest, 'avg_step_ms'):.2f} ms)",
      f"{flattest['optimizer']} (cv {as_float(flattest, 'row_norm_cv'):.3f})",
      f"{geometry['optimizer']} ({as_float(geometry, 'orthogonality_defect'):.3f})",
      f"{aligned['optimizer']} ({as_float(aligned, 'gradient_alignment'):.3f})",
    )

  return t


def build_scoreboard(s: State) -> Table:
  t = Table(
    title=f"[bold {WHITE}]SCOREBOARD[/]",
    border_style=DARK,
    box=box.SIMPLE,
    expand=False,
    show_header=True,
    pad_edge=False,
    title_justify="left",
  )
  t.add_column("optimizer", style=LIGHT, no_wrap=True)
  t.add_column("total wins", justify="right", width=10)
  t.add_column("where it wins", style=MID)

  wins = wins_by_shape(s.rows)
  score: dict[str, list[str]] = {}
  for shape, metrics in wins.items():
    for metric, optimizers in metrics.items():
      for opt in optimizers:
        score.setdefault(opt, []).append(f"{shape}:{metric}")

  for opt, labels in sorted(score.items(), key=lambda item: (-len(item[1]), item[0])):
    t.add_row(opt, str(len(labels)), ", ".join(labels[:8]) + (" ..." if len(labels) > 8 else ""))

  return t


def build_footer(s: State) -> Text:
  parts: list[tuple[str, str]] = [(f"{s.elapsed_s():.1f}s", LIGHT)]
  if s.done:
    parts.append(("done", f"bold {WHITE}"))
  elif s.current_optimizer and s.current_shape:
    parts.append((f"{s.current_optimizer} {s.current_shape}", MID))
  out = Text()
  for i, (text, style) in enumerate(parts):
    if i:
      out.append(" - ", style=DIM)
    out.append(text, style=style)
  return out


class TUI:
  def __init__(self, state: State):
    self.s = state

  def __rich__(self) -> Group:
    parts: list[object] = [build_table(self.s), Text("")]
    if self.s.rows:
      parts += [build_summary(self.s), Text(""), build_scoreboard(self.s), Text("")]
    parts.append(build_footer(self.s))
    return Group(*parts)


def parse_csv_row(line: str) -> dict[str, str] | None:
  if line.startswith("optimizer,"):
    return None
  try:
    rows = list(csv.DictReader(
      ["optimizer,N,M,avg_step_ms,update_fro_norm,row_norm_mean,row_norm_std,row_norm_cv,dead_row_fraction,orthogonality_defect,gradient_alignment,row_norm_min,row_norm_max", line]
    ))
  except csv.Error:
    return None
  return rows[0] if rows else None


def main() -> int:
  console = Console()
  ensure_binary(console)
  state = State()

  proc = subprocess.Popen(
    [str(BIN)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    text=True,
    cwd=str(ROOT),
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
  )

  with Live(TUI(state), console=console, refresh_per_second=10, transient=False):
    assert proc.stdout is not None
    for raw in proc.stdout:
      line = raw.strip()
      if not line:
        continue
      if line.startswith("PROGRESS,"):
        _, opt, n, m = line.split(",", 3)
        state.current_optimizer = opt
        state.current_shape = f"{n}x{m}"
        continue
      row = parse_csv_row(line)
      if row is None:
        state.unknown_lines.append(line)
        continue
      state.rows.append(row)
    state.done = True

  rc = proc.wait()
  if rc != 0:
    console.print(f"[{WHITE}]exited rc={rc}[/]")
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
