"""preprocess.py
Synthetic data generation utilities used in the refactored project.  The real
ImageNet dataset is far too large for the execution environment, therefore we
retain the lightweight random-tensor dataset from the original script.
"""

from __future__ import annotations
import random
from typing import Tuple

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise RuntimeError("PyTorch is required for data handling: " + str(e))

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("numpy is required: " + str(e))

# ----------------------------------------------------------------------------------
# 1)  Global seed for reproducible synthetic data
# ----------------------------------------------------------------------------------
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)           # type: ignore
torch.manual_seed(SEED)        # type: ignore

# ----------------------------------------------------------------------------------
# 2)  Synthetic ImageNet-shaped dataset & loader helper
# ----------------------------------------------------------------------------------
class SyntheticDataset(Dataset):
    """Tiny random-tensor dataset that mimics ImageNet images & labels."""

    def __init__(self, num_examples: int = 256):
        self.X = torch.rand(num_examples, 3, 224, 224)
        self.y = torch.randint(0, 1000, (num_examples,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_loader(bs: int = 64, n: int = 256) -> DataLoader:
    ds = SyntheticDataset(n)
    return DataLoader(ds, batch_size=bs, shuffle=False)
