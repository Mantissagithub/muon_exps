import torch
from torch.optim import Muon
import time

device = torch.device("cuda")
sizes = [[1024, 1024], [2048, 2048], [4096, 4096]]
for N, M in sizes:
  W = torch.randn(N, M, device=device, requires_grad=True)
  optimizer = Muon([W], lr=1e-3, weight_decay=0.1)

  torch.cuda.synchronize()
  start = time.time()
  for _ in range(1000):
      loss = (W ** 2).sum()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
  torch.cuda.synchronize()
  print(f"PyTorch Muon: {(time.time() - start)/1000*1000:.2f} ms/step")