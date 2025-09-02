"""src/preprocess.py
Data-loading utilities and common helpers extracted from the original
experiment script.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

__all__ = [
    "set_seed",
    "MemThroughputProfiler",
    "build_imagenet_loader",
]

# ---------------------------------------------------------------------------
#  REPRODUCIBILITY
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Deterministic seed across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  LIGHT-WEIGHT PROFILER
# ---------------------------------------------------------------------------

class MemThroughputProfiler:
    """Profiler that tracks peak GPU memory and throughput.

    Works gracefully on CPU-only environments by returning zeros for memory.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        self.n_items = 0

    def update(self, batch_size: int):
        self.n_items += batch_size

    def summary(self) -> Dict[str, float]:
        elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 2 ** 20  # MiB
        else:
            peak_mem = 0.0
        ips = self.n_items / elapsed
        return {"peak_mem_MiB": round(peak_mem, 2), "img_per_sec": round(ips, 3)}


# ---------------------------------------------------------------------------
#  IMAGENET DATALOADER (identical logic to the original script)
# ---------------------------------------------------------------------------

def build_imagenet_loader(
    res: int,
    batch_size: int,
    split: str = "train",
    synthetic: bool = False,
) -> DataLoader:
    """Return a synthetic or real ImageNet(-like) *DataLoader*.

    If *synthetic* is ``True`` an in-memory random dataset is used to avoid
    costly disk I/O, mirroring the original script's behaviour.
    """

    if synthetic:

        class _SynthDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 12800  # arbitrary large number

            def __getitem__(self, idx):  # noqa: D401 – keep compatible
                img = torch.randint(0, 256, (3, res, res), dtype=torch.uint8)
                label = torch.randint(0, 1000, (1,)).item()
                img = img.float() / 255.0
                return img, label

        ds = _SynthDataset()
        return DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
        )

    # ---------------------------------------------------------------------
    # Real ImageNet loader (path is resolved from the DATA env variable)
    # ---------------------------------------------------------------------
    root = os.environ.get("DATA", "./data")
    root = Path(root) / "imagenet" / ("train" if split == "train" else "val")

    tfms = transforms.Compose(
        [
            transforms.Resize(int(res * 1.05)),
            transforms.CenterCrop(res),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    from torchvision.datasets import ImageFolder  # local import to save RAM

    ds = ImageFolder(root=str(root), transform=tfms)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)