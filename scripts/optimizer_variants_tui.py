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


def fmt_num(row: dict[str, str], key: str, digits: int = 3) -> str:
  try:
    return f"{float(row[key]):.{digits}f}"
  except (KeyError, ValueError):
    return "-"


def build_table(s: State) -> Table:
  t = Table(
    title=f"[bold {WHITE}]OPTIMIZER VARIANTS[/]  [{MID}]synthetic anisotropic update benchmark[/]",
    border_style=DARK,
    box=box.ROUNDED,
    expand=False,
    show_lines=False,
    pad_edge=False,
    title_justify="left",
  )
  t.add_column("optimizer", style=LIGHT, no_wrap=True)
  t.add_column("shape", style=MID, no_wrap=True)
  t.add_column("ms", justify="right", width=9)
  t.add_column("row cv", justify="right", width=9)
  t.add_column("dead", justify="right", width=8)
  t.add_column("ortho", justify="right", width=9)
  t.add_column("align", justify="right", width=9)
  t.add_column("row min/max", justify="right", width=17)

  for row in s.rows:
    shape = f"{row['N']}x{row['M']}"
    minmax = f"{fmt_num(row, 'row_norm_min')}/{fmt_num(row, 'row_norm_max')}"
    t.add_row(
      row["optimizer"],
      shape,
      fmt_num(row, "avg_step_ms"),
      fmt_num(row, "row_norm_cv"),
      fmt_num(row, "dead_row_fraction"),
      fmt_num(row, "orthogonality_defect"),
      fmt_num(row, "gradient_alignment"),
      minmax,
    )

  if not s.done and s.current_optimizer and s.current_shape:
    t.add_row(
      Text(s.current_optimizer, style=DIM),
      Text(s.current_shape, style=DIM),
      cell_running(),
      Text("-", style=DIM),
      Text("-", style=DIM),
      Text("-", style=DIM),
      Text("-", style=DIM),
      Text("-", style=DIM),
    )

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
    return Group(build_table(self.s), build_footer(self.s))


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

  rc = proc.wait()
  state.done = True
  if rc != 0:
    console.print(f"[{WHITE}]exited rc={rc}[/]")
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
