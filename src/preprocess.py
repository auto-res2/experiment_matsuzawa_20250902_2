"""src/preprocess.py
Data-loading and preprocessing logic.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

__all__ = ["build_loader"]


def build_loader(
    batch_size: int,
    img_size: int = 224,
    num_samples: int = 1024,
) -> DataLoader:
    """Return a DataLoader based on torchvision's FakeData used as ImageNet stand-in."""

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
