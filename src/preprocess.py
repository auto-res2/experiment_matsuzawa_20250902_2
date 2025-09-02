"""src/preprocess.py
Data loading & augmentation utilities. Uses ImageNet/-C/-A roots when available
and otherwise falls back to a tiny in-memory synthetic set so that CI executes
quickly even without the large datasets.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------------
#  Dataset roots (override via environment variables if desired)
# ----------------------------------------------------------------------------------
IMAGENET_ROOT = Path("/mnt/imagenet")
IMAGENET100_ROOT = Path("/mnt/imagenet100")
IMAGENET_C_ROOT = Path("/mnt/imagenet_c")
IMAGENET_A_ROOT = Path("/mnt/imagenet_a")

# ----------------------------------------------------------------------------------
#  Synthetic tiny fallback dataset – keeps CI < 10 s
# ----------------------------------------------------------------------------------

class TinySynthetic(torch.utils.data.Dataset):
    """20 random 224 × 224 RGB images across 10 classes."""

    def __init__(self, n_class: int = 10, n_img: int = 20):
        self.data = torch.randn(n_img, 3, 224, 224)
        self.label = torch.randint(0, n_class, (n_img,))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return self.data[i], self.label[i]

# ----------------------------------------------------------------------------------
#  Transforms (ImageNet standard)
# ----------------------------------------------------------------------------------
TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
        transforms.RandAugment(2, 9),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

TRANSFORM_VAL = transforms.Compose(
    [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ----------------------------------------------------------------------------------
#  Loader builder with graceful degradation
# ----------------------------------------------------------------------------------

def _build_loader(root: Path, train: bool, batch: int, synthetic_ok: bool = True):
    """Construct a dataloader. Falls back to `TinySynthetic` when the real
    dataset tree is absent so that public CI can still run all code paths.
    """
    if root.exists():
        ds_root = root / ("train" if train else "val")
        if ds_root.exists():
            ds = datasets.ImageFolder(ds_root, TRANSFORM_TRAIN if train else TRANSFORM_VAL)
        elif (not train) and root.exists():  # some 100-class splits are flat
            ds = datasets.ImageFolder(root, TRANSFORM_VAL)
        else:
            raise RuntimeError(f"Dataset split not found under {root}")
    elif synthetic_ok:
        warnings.warn(
            f"Falling back to TinySynthetic dataset – {root} missing; results will not be valid"
        )
        ds = TinySynthetic()
    else:
        raise RuntimeError(f"Dataset path {root} missing and synthetic_ok=False")

    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=train,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
