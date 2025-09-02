"""src/preprocess.py
Dataset locations, augmentation pipelines and data-loader helpers.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# 0.  DATA LOCATIONS -----------------------------------------------------------
# -----------------------------------------------------------------------------
DATA_ROOTS = {
    "imagenet": Path("/mnt/imagenet"),
    "imagenet_c": Path("/mnt/imagenet_c"),
    "imagenet_a": Path("/mnt/imagenet_a"),
    "imagenet100": Path("/mnt/imagenet100"),
}

# ImageNet normalisation
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Training / Validation augmentation policies ---------------------------------
TRAIN_AUG = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
        transforms.RandAugment(2, 9),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*IMAGENET_STATS),
    ]
)

VAL_AUG = transforms.Compose(
    [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*IMAGENET_STATS),
    ]
)

# -----------------------------------------------------------------------------
# 1.  DATA-LOADER BUILDERS -----------------------------------------------------
# -----------------------------------------------------------------------------

def build_loader(name: str, split: str, batch: int, workers: int = 8) -> Tuple[DataLoader, int]:
    """Return (DataLoader, dataset-length). The function aborts with `FileNotFoundError`
    if the requested dataset directory is missing.
    """

    root = DATA_ROOTS[name]
    if not root.exists():
        raise FileNotFoundError(f"Required dataset directory '{root}' not found.")

    is_train = split == "train"
    trans = TRAIN_AUG if is_train else VAL_AUG

    # ImageNet classic has train/val sub-dirs; corruption & others already split.
    if name == "imagenet":
        sub = "train" if is_train else "val"
        path = root / sub
    else:
        path = root

    ds = datasets.ImageFolder(path, transform=trans)
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=is_train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(ds)
