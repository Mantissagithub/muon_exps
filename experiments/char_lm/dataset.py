from pathlib import Path
from urllib.request import urlretrieve

import torch


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharDataset:
    def __init__(self, text: str, block_size: int, split: float = 0.9):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size

        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        n = int(split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, batch_size: int, device: torch.device, generator: torch.Generator):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,), generator=generator)
        x = torch.stack([data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix]).to(device)
        return x, y


def load_tinyshakespeare(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")
