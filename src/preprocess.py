"""src/preprocess.py
Data-loading & preprocessing utilities.
The original script uses `torchvision.datasets.FakeData` so we keep that logic.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

# ---------------------------------------------------------------------------- #
# Data loader builder
# ---------------------------------------------------------------------------- #

def build_loader(batch_size: int, num_samples: int = 2048, img_size: int = 224):
    """Return a DataLoader over a FakeData ImageNet-like dataset."""

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = FakeData(
        size=num_samples,
        image_size=(3, img_size, img_size),
        num_classes=1000,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader
