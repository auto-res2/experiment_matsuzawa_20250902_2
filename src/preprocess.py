"""src/preprocess.py
Data-loading / preprocessing helpers shared by the experiments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
#  ImageNet loader used by Experiment-1
# -----------------------------------------------------------------------------

def get_imagenet_loader(
    data_dir: str | None = None,
    img_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Return val-set DataLoader for ImageNet-1k.
    The directory can be supplied via argument or the env-var `IMAGENET_VAL_DIR`.
    Raises FileNotFoundError if the folder is missing.
    """
    import torchvision.datasets as dsets  # local import

    root = data_dir or os.getenv("IMAGENET_VAL_DIR", None)
    if root is None or not Path(root).exists():
        raise FileNotFoundError("Imagenet validation directory not found.")

    transform = T.Compose([
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    ds = dsets.ImageFolder(root=root, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# -----------------------------------------------------------------------------
#  Synthetic patch dataset for Experiment-3
# -----------------------------------------------------------------------------

def patch_dataset(
    n: int = 1_000,
    base_size: int = 512,
    patch_size: int = 128,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (images, masks) tensors for the synthetic checker-board patch test."""
    rng = np.random.default_rng(seed)
    imgs, masks = [], []
    for _ in range(n):
        img = rng.random((3, base_size, base_size), dtype=np.float32)
        mask = np.zeros((1, base_size, base_size), dtype=np.float32)
        x = rng.integers(0, base_size - patch_size)
        y = rng.integers(0, base_size - patch_size)
        checker = np.tile(((np.indices((patch_size, patch_size)).sum(axis=0) % 2).astype(np.float32)[None]), (3, 1, 1))
        img[:, y : y + patch_size, x : x + patch_size] = checker
        mask[:, y : y + patch_size, x : x + patch_size] = 1.0
        imgs.append(torch.from_numpy(img))
        masks.append(torch.from_numpy(mask))

    return torch.stack(imgs), torch.stack(masks)
