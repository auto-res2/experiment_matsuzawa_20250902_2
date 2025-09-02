"""src/preprocess.py
Dataset downloading & preprocessing helpers extracted from the monolithic script.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = Path("data")

# --------------------------------------------------------------------
#  Waterbirds
# --------------------------------------------------------------------

def prepare_waterbirds() -> Dict[str, Dataset]:
    """Download Waterbirds-95 splits via 🤗 Datasets and wrap into torch Dataset."""
    name = "grodino/waterbirds"
    ds = load_dataset(name)
    assert {"train", "validation", "test"}.issubset(ds.keys()), "Waterbirds splits missing!"

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class _Wrapped(Dataset):
        def __init__(self, hf_split):
            self.d = hf_split
            self.tf = transform

        def __len__(self):
            return len(self.d)

        def __getitem__(self, idx):
            sample = self.d[idx]
            img = sample["image"]
            y = int(sample["y"])
            env = int(sample["place"])
            return self.tf(img), y, env

    return {k: _Wrapped(v) for k, v in ds.items()}

# --------------------------------------------------------------------
#  CelebA (hair-colour task, spurious attr = sex)
# --------------------------------------------------------------------

def _read_celeba_attr(root: Path) -> pd.DataFrame:
    attr_file = root / "list_attr_celeba.txt"
    # The first line is the number of images; skip it
    df = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1)
    df.index = df["image_id"]
    df.drop(columns=["image_id"], inplace=True)
    df.replace({-1: 0}, inplace=True)  # map -1 ➔ 0
    return df


def prepare_celeba() -> Dict[str, Dataset]:
    """CelebA hair-colour binary task; spurious attribute is *Male*."""
    root = DATA_DIR / "celeba"
    root.mkdir(parents=True, exist_ok=True)

    # Ensure availability (torchvision handles download / checksum)
    _ = tv.datasets.CelebA(root=str(root), split="all", download=True)

    attr_df = _read_celeba_attr(root)
    part_df = pd.read_csv(
        root / "list_eval_partition.txt", delim_whitespace=True, names=["image_id", "split"]
    )

    split_map = {0: "train", 1: "validation", 2: "test"}
    target_attr, spurious_attr = "Blond_Hair", "Male"

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    splits: Dict[str, Dataset] = {}
    for s_id, split_name in split_map.items():
        file_names = part_df[part_df["split"] == s_id]["image_id"].tolist()

        class _CelebSubset(Dataset):
            def __len__(self):
                return len(file_names)

            def __getitem__(self, i):
                fname = file_names[i]
                img_path = root / "img_align_celeba" / fname
                img = Image.open(img_path).convert("RGB")
                y = int(attr_df.loc[fname, target_attr])
                env = int(attr_df.loc[fname, spurious_attr])
                return transform(img), y, env

        splits[split_name] = _CelebSubset()

    return splits

# --------------------------------------------------------------------
#  CIFAR-Spurious (coloured patch)
# --------------------------------------------------------------------

def prepare_cifar_spurious() -> Dict[str, Dataset]:
    """CIFAR-Spurious: coloured corner patch present in 95 % of training images."""

    patch_prob = 0.95
    patch_size = 8
    num_classes = 10

    def add_patch(img: Image.Image, label: int) -> Image.Image:
        img = img.copy()
        colour_map = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 128, 0),
            (128, 0, 128),
            (0, 128, 128),
            (64, 64, 64),
        ]
        rgb = colour_map[label]
        for i in range(patch_size):
            for j in range(patch_size):
                img.putpixel((j, i), rgb)
        return img

    transform_common = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ---------------- create splits ---------------- #
    base_train = tv.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    base_test = tv.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)

    # Validation = first 5k images of test set, Test = remaining 5k
    idx_val, idx_test = list(range(5000)), list(range(5000, 10000))

    def make(base_dataset, indices, train_flag: bool):
        class _CIFARSpurious(Dataset):
            def __len__(self):
                return len(indices)

            def __getitem__(self, idx):
                img, y = base_dataset[indices[idx]]
                if train_flag and random.random() < patch_prob:
                    img = add_patch(img, y)
                    env = 1
                else:
                    env = 0
                return transform_common(img), y, env

        return _CIFARSpurious()

    return {
        "train": make(base_train, list(range(len(base_train))), True),
        "validation": make(base_test, idx_val, False),
        "test": make(base_test, idx_test, False),
    }
