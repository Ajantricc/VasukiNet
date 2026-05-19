from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CharBlockDataset(Dataset):
    def __init__(self, ids: list[int], block_size: int):
        if len(ids) <= block_size + 1:
            raise ValueError("Dataset is too small for the selected block_size.")
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        chunk = self.data[idx: idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]
