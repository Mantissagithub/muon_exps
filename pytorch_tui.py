# run via: uv run pytorch_tui.py
# compact rich tui for the pytorch muon baseline. grey + white only.

import threading
import time
from pathlib import Path

import torch
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


HERE = Path(__file__).parent.resolve()

WHITE = "white"
LIGHT = "grey85"
MID = "grey70"
DARK = "grey50"
DIM = "grey35"

# same 6 shapes as benchmark.cu so rows line up with the cuda tui
SHAPES = [
  (1024, 1024),
  (2048, 2048),
  # (4096, 4096),
  (4096, 1024),
  # (8192, 2048),
  (8192, 1024),
]
ITERS = 1000


class State:
  def __init__(self) -> None:
    self.rows: list[dict] = []
    self.t0: float = time.monotonic()
    self.running_idx: int | None = None
    self.done: bool = False
    self.device: str = ""
    self.cuda_ver: str = ""

  def elapsed_s(self) -> float:
    return time.monotonic() - self.t0


def cell_running() -> Spinner:
  return Spinner("dots", text=Text("running", style=DIM), style=DIM)


def build_table(s: State) -> Table:
  t = Table(
    title=f"[bold {WHITE}]PYTORCH MUON BASELINE[/]  [{MID}]· torch.optim.Muon[/]",
    border_style=DARK,
    box=box.ROUNDED,
    expand=False,
    show_lines=False,
    pad_edge=False,
    title_justify="left",
  )
  t.add_column("shape", style=LIGHT, no_wrap=True)
  t.add_column("ρ", style=MID, justify="right", width=4)
  t.add_column("ms/step", justify="right", width=14)

  for r in s.rows:
    is_running = s.running_idx == r["idx"]
    if r["ms"] is not None:
      ms_cell: object = Text(f"{r['ms']:.2f} ms", style=LIGHT)
    elif is_running:
      ms_cell = cell_running()
    else:
      ms_cell = Text("—", style=DIM)
    t.add_row(f"{r['N']}×{r['M']}", f"{r['rho']:.1f}", ms_cell)

  return t


def build_footer(s: State) -> Text:
  parts: list[tuple[str, str]] = []
  if s.device:
    parts.append((s.device, LIGHT))
  if s.cuda_ver:
    parts.append((f"cuda {s.cuda_ver}", MID))
  parts.append((f"iters {ITERS}", MID))
  parts.append((f"{s.elapsed_s():.1f}s", LIGHT))
  if s.done:
    parts.append(("done", f"bold {WHITE}"))
  elif s.running_idx is not None:
    parts.append((f"shape {s.running_idx}/{len(SHAPES)}", MID))
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


def bench_one(N: int, M: int, device: torch.device) -> float:
  W = torch.randn(N, M, device=device, requires_grad=True)
  optimizer = torch.optim.Muon([W], lr=1e-3, weight_decay=0.1)

  # warmup
  loss = (W ** 2).sum(); loss.backward(); optimizer.step(); optimizer.zero_grad()

  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(ITERS):
    loss = (W ** 2).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  torch.cuda.synchronize()
  ms = (time.perf_counter() - start) / ITERS * 1000.0
  del W, optimizer
  torch.cuda.empty_cache()
  return ms


def run_bench(state: State, device: torch.device) -> None:
  for idx, (N, M) in enumerate(SHAPES, start=1):
    big, small = max(N, M), min(N, M)
    state.rows.append({"idx": idx, "N": N, "M": M, "rho": big / small, "ms": None})
    state.running_idx = idx
    ms = bench_one(N, M, device)
    state.rows[-1]["ms"] = ms
  state.running_idx = None
  state.done = True


def main() -> int:
  console = Console()

  if not torch.cuda.is_available():
    console.print(f"[{WHITE}]cuda not available[/]")
    return 1

  device = torch.device("cuda")
  state = State()
  state.device = torch.cuda.get_device_name(0)
  state.cuda_ver = torch.version.cuda or "?"

  worker = threading.Thread(target=run_bench, args=(state, device), daemon=True)

  with Live(TUI(state), console=console, refresh_per_second=12, transient=False):
    worker.start()
    worker.join()

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
