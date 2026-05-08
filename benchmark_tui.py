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

# fp32 variant tags emitted by benchmark.cu, in display order
GNS_VARIANTS = ["quintic", "polar", "polar_restart", "polar_restart_syrk"]
HEADER_LABELS = {
  "quintic": "Quintic",
  "polar": "Polar",
  "polar_restart": "+Restart",
  "polar_restart_syrk": "+Syrk",
}


def ensure_binary(console: Console) -> None:
  if BIN.exists():
    return
  console.print(f"[{MID}]benchmark binary not found, compiling…[/]")
  if shutil.which("nvcc") is None:
    console.print(f"[{WHITE}]nvcc not on PATH — install CUDA toolkit[/]")
    sys.exit(1)
  rc = subprocess.call(["nvcc", "-o", "benchmark", "benchmark.cu", "-lcublas", "-arch=sm_89"], cwd=str(HERE))
  if rc != 0 or not BIN.exists():
    console.print(f"[{WHITE}]compile failed[/]")
    sys.exit(1)


class State:
  def __init__(self) -> None:
    self.sysinfo: dict = {}
    self.rows: list[dict] = []
    self.verifies: list[dict] = []
    self.verifying: bool = False
    self.total: int = 0
    self.t0: float = time.monotonic()
    self.running_idx: int | None = None
    self.running_variant: str | None = None
    self.done: bool = False
    self.unknown_lines: list[str] = []  # stderr/stdout lines not matching any tag

  def elapsed_s(self) -> float:
    return time.monotonic() - self.t0


def cell_running() -> Spinner:
  return Spinner("dots", text=Text("running", style=DIM), style=DIM)


def build_verify_table(s: State) -> Table:
  t = Table(
    title=f"[bold {WHITE}]VERIFY[/]  [{MID}]· Gram NS vs v1 NS  ·  fp16 vs fp32 polar+restart[/]",
    border_style=DARK,
    box=box.ROUNDED,
    expand=False,
    show_lines=False,
    pad_edge=False,
    title_justify="left",
  )
  t.add_column("shape", style=LIGHT, no_wrap=True)
  t.add_column("mode", style=MID, no_wrap=True)
  t.add_column("max diff", style=LIGHT, justify="right", width=12)
  t.add_column("status", justify="right", width=6)

  if s.verifying and not s.verifies:
    t.add_row(Text("running", style=DIM), Text("—", style=DIM), Text("—", style=DIM), cell_running())
  else:
    for v in s.verifies:
      ok = v["status"] == "ok"
      st = Text("ok" if ok else "fail", style=(f"bold {WHITE}" if ok else f"bold {WHITE}"))
      t.add_row(f"{v['N']}×{v['M']}", v["mode"], f"{v['diff']:.2e}", st)

  return t


def build_bench_table(s: State) -> Table:
  t = Table(
    title=f"[bold {WHITE}]MUON CUDA BENCHMARK[/]  [{MID}]· v1 NS vs stabilized Gram NS[/]",
    border_style=DARK,
    box=box.ROUNDED,
    expand=False,
    show_lines=False,
    pad_edge=False,
    title_justify="left",
  )
  t.add_column("shape", style=LIGHT, no_wrap=True)
  t.add_column("ρ", style=MID, justify="right", width=4)
  t.add_column("v1 NS", justify="right", width=10)
  for v in GNS_VARIANTS:
    t.add_column(HEADER_LABELS[v], justify="right", width=10)
  t.add_column("best", justify="right", width=10)

  for r in s.rows:
    is_running = s.running_idx == r["idx"]
    cells: list[object] = [f"{r['N']}×{r['M']}", f"{r['rho']:.1f}"]

    def fmt(key: str) -> object:
      val = r.get(key)
      if val is not None:
        return Text(f"{val:.2f}", style=LIGHT)
      if is_running and s.running_variant == key:
        return cell_running()
      return Text("—", style=DIM)

    cells.append(fmt("v1_ns"))
    for v in GNS_VARIANTS:
      cells.append(fmt(v))

    v1 = r.get("v1_ns")
    gns_done = [r.get(v) for v in GNS_VARIANTS if r.get(v) is not None]
    if v1 is not None and gns_done:
      best = min(gns_done)
      spd = v1 / best
      arrow = "▲" if spd >= 1.0 else "▼"
      cells.append(Text(f"{arrow} {spd:.2f}×", style=f"bold {WHITE}"))
    else:
      cells.append(Text("—", style=DIM))

    t.add_row(*cells)

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
  elif s.verifying:
    parts.append(("verifying", MID))
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
    return Group(build_verify_table(self.s), Text(""), build_bench_table(self.s), build_footer(self.s))


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
      elif tag == "VERIFY_START":
        state.verifying = True
      elif tag == "VERIFY_END":
        state.verifying = False
      elif tag == "VERIFY" and len(parts) >= 6:
        N, M, mode, diff, status = int(parts[1]), int(parts[2]), parts[3], float(parts[4]), parts[5]
        state.verifies.append({"N": N, "M": M, "mode": mode, "diff": diff, "status": status})
      elif tag == "TOTAL" and len(parts) >= 2:
        state.total = int(parts[1])
      elif tag == "START" and len(parts) >= 5:
        idx, N, M, rho = int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
        row: dict = {"idx": idx, "N": N, "M": M, "rho": rho, "v1_ns": None}
        for v in GNS_VARIANTS:
          row[v] = None
        state.rows.append(row)
        state.running_idx = idx
        state.running_variant = "v1_ns"
      elif tag == "TIME" and len(parts) >= 4:
        idx, variant, ms = int(parts[1]), parts[2], float(parts[3])
        for r in state.rows:
          if r["idx"] == idx:
            r[variant] = ms
            break
        order = ["v1_ns"] + GNS_VARIANTS
        if variant in order:
          i = order.index(variant)
          state.running_variant = order[i+1] if i+1 < len(order) else None
      elif tag == "END":
        state.running_variant = None
      elif tag == "DONE":
        state.running_idx = None
        state.running_variant = None
        state.done = True
      else:
        state.unknown_lines.append(line)

  rc = proc.wait()
  if rc != 0:
    console.print(f"[{WHITE}]exited rc={rc}[/]")
  if state.unknown_lines and (rc != 0 or os.environ.get("BENCH_TUI_VERBOSE")):
    console.print(f"[{DIM}]subprocess output (non-tag lines):[/]")
    for ln in state.unknown_lines:
      console.print(f"[{MID}]{ln}[/]")
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
