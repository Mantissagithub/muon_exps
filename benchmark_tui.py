# run via: uv run benchmark_tui.py
# compact rich tui wrapper for ./benchmark. grey + white only.

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


HERE = Path(__file__).parent.resolve()
BIN = HERE / "benchmark"

WHITE = "white"
LIGHT = "grey85"
MID = "grey70"
DARK = "grey50"
DIM = "grey35"


def ensure_binary(console: Console) -> None:
  if BIN.exists():
    return
  console.print(f"[{MID}]benchmark binary not found, compiling…[/]")
  if shutil.which("nvcc") is None:
    console.print(f"[{WHITE}]nvcc not on PATH — install CUDA toolkit[/]")
    sys.exit(1)
  rc = subprocess.call(["nvcc", "-o", "benchmark", "benchmark.cu", "-lcublas"], cwd=str(HERE))
  if rc != 0 or not BIN.exists():
    console.print(f"[{WHITE}]compile failed[/]")
    sys.exit(1)


class State:
  def __init__(self) -> None:
    self.sysinfo: dict = {}
    self.rows: list[dict] = []
    self.total: int = 0
    self.t0: float = time.monotonic()
    self.running_idx: int | None = None
    self.running_variant: str | None = None
    self.done: bool = False

  def elapsed_s(self) -> float:
    return time.monotonic() - self.t0


def cell_running() -> Spinner:
  return Spinner("dots", text=Text("running", style=DIM), style=DIM)


def build_table(s: State) -> Table:
  t = Table(
    title=f"[bold {WHITE}]MUON CUDA BENCHMARK[/]  [{MID}]· v1 NS vs v1 Gram NS[/]",
    border_style=DARK,
    box=box.ROUNDED,
    expand=False,
    show_lines=False,
    pad_edge=False,
    title_justify="left",
  )
  t.add_column("shape", style=LIGHT, no_wrap=True)
  t.add_column("ρ", style=MID, justify="right", width=4)
  t.add_column("v1 NS", justify="right", width=14)
  t.add_column("v1 Gram NS", justify="right", width=14)
  t.add_column("speedup", justify="right", width=10)

  for r in s.rows:
    idx = r["idx"]
    is_running = s.running_idx == idx
    if r["v1_ns"] is not None:
      v1: object = Text(f"{r['v1_ns']:.2f} ms", style=LIGHT)
    elif is_running and s.running_variant == "v1_ns":
      v1 = cell_running()
    else:
      v1 = Text("—", style=DIM)
    if r["gns"] is not None:
      gn: object = Text(f"{r['gns']:.2f} ms", style=LIGHT)
    elif is_running and s.running_variant == "gns":
      gn = cell_running()
    else:
      gn = Text("—", style=DIM)
    if r["v1_ns"] is not None and r["gns"] is not None:
      spd = r["v1_ns"] / r["gns"]
      arrow = "▲" if spd >= 1.0 else "▼"
      spd_cell: object = Text(f"{arrow} {spd:.2f}×", style=f"bold {WHITE}")
    else:
      spd_cell = Text("—", style=DIM)
    t.add_row(f"{r['N']}×{r['M']}", f"{r['rho']:.1f}", v1, gn, spd_cell)

  return t


def build_footer(s: State) -> Text:
  parts: list[tuple[str, str]] = []
  if s.sysinfo:
    parts += [
      (s.sysinfo.get("device", "?"), LIGHT),
      ("cc " + s.sysinfo.get("cc", "?"), MID),
      (s.sysinfo.get("vram", "?"), LIGHT),
      ("iters " + s.sysinfo.get("iters", "?"), MID),
    ]
  parts.append((f"{s.elapsed_s():.1f}s", LIGHT))
  if s.done:
    parts.append(("done", f"bold {WHITE}"))
  elif s.running_idx is not None:
    parts.append((f"shape {s.running_idx}/{s.total} · {s.running_variant or '...'}", MID))
  out = Text()
  for i, (text, style) in enumerate(parts):
    if i:
      out.append(" · ", style=DIM)
    out.append(text, style=style)
  return out


class TUI:
  def __init__(self, state: State):
    self.s = state
  def __rich__(self) -> Group:
    return Group(build_table(self.s), build_footer(self.s))


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
    cwd=str(HERE),
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
  )

  with Live(TUI(state), console=console, refresh_per_second=12, transient=False):
    assert proc.stdout is not None
    for raw in proc.stdout:
      line = raw.rstrip()
      if not line:
        continue
      parts = line.split("|")
      tag = parts[0]
      if tag == "SYS" and len(parts) >= 6:
        state.sysinfo = {
          "device": parts[1],
          "cc": parts[2],
          "vram": f"{parts[3]} GiB",
          "cuda_rt": parts[4],
          "iters": parts[5],
        }
      elif tag == "TOTAL" and len(parts) >= 2:
        state.total = int(parts[1])
      elif tag == "START" and len(parts) >= 5:
        idx, N, M, rho = int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
        state.rows.append({"idx": idx, "N": N, "M": M, "rho": rho, "v1_ns": None, "gns": None})
        state.running_idx = idx
        state.running_variant = "v1_ns"
      elif tag == "TIME" and len(parts) >= 4:
        idx, variant, ms = int(parts[1]), parts[2], float(parts[3])
        for r in state.rows:
          if r["idx"] == idx:
            r[variant] = ms
            break
        state.running_variant = "gns" if variant == "v1_ns" else None
      elif tag == "END":
        state.running_variant = None
      elif tag == "DONE":
        state.running_idx = None
        state.running_variant = None
        state.done = True

  rc = proc.wait()
  if rc != 0:
    console.print(f"[{WHITE}]exited rc={rc}[/]")
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
