"""src/preprocess.py
Dataset creation, transforms and high-level task splitting utilities.
"""
from __future__ import annotations
from typing import List, Tuple
import random
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision import transforms, datasets

# -----------------------------------------------------------------------------
#  Global transforms (224×224 to match ResNet-18 default crop size)
# -----------------------------------------------------------------------------

NORMALISE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

BASIC_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    NORMALISE,
])

AUG_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    NORMALISE,
])

# -----------------------------------------------------------------------------
#  Generic N-way class split helper
# -----------------------------------------------------------------------------

def split_dataset(dataset: datasets.VisionDataset, class_order: List[int], split_size: int) -> List[Subset]:
    tasks: List[Subset] = []
    for i in range(0, len(class_order), split_size):
        cls = class_order[i: i + split_size]
        idx = [j for j, y in enumerate(getattr(dataset, "targets")) if y in cls]
        tasks.append(Subset(dataset, idx))
    return tasks

# -----------------------------------------------------------------------------
#  CIFAR-100 split-10×10 as used in the paper
# -----------------------------------------------------------------------------

def get_split_cifar100(*, root: str | Path = "./data") -> Tuple[List[Subset], List[Subset]]:
    class_order = list(range(100))
    random.shuffle(class_order)
    train_ds = datasets.CIFAR100(root=root, train=True, download=True, transform=AUG_TRANSFORM)
    test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=BASIC_TRANSFORM)
    train_tasks = split_dataset(train_ds, class_order, 10)
    test_tasks  = split_dataset(test_ds,  class_order, 10)
    return train_tasks, test_tasks

# -----------------------------------------------------------------------------
#  (Optional) Rotated-MNIST stub – real implementation is in supplementary repo
# -----------------------------------------------------------------------------

from torchvision.datasets import MNIST

class RotatedMNIST(MNIST):
    def __init__(self, *a, rotation: float = 0.0, **kw):
        base_transform = kw.get("transform", transforms.ToTensor())
        super().__init__(*a, transform=base_transform, **kw)
        self.rotation = rotation

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x = transforms.functional.rotate(x, self.rotation)
        return x, y
