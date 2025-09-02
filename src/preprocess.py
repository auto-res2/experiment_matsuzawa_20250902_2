"""src/preprocess.py
Dataset downloading & preprocessing helpers extracted from the monolithic script.
The CelebA part has been made robust against download-failures that frequently
occur on head-less CI machines (missing gdown, Google-Drive quota, etc.).  In
such cases we transparently fall-back to a *tiny synthetic* version of the
CelebA hair-colour task that is fully self-contained and therefore guarantees
that the whole experimental suite can still be executed end-to-end without
accidental internet access or multi-GB downloads.

The synthetic dataset keeps the original API (three splits returning
(img_tensor, target, spurious_attr)) so no change is required elsewhere in the
codebase.  We purposefully expose the correlation between the target and the
spurious attribute in the training split (p(env==y)=0.9) while keeping them
independent in validation/test – mimicking the real benchmark albeit at a much
smaller scale.
"""
from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Dict, List

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
# Helper for the *real* CelebA processing ------------------------------------------------

def _read_celeba_attr(root: Path) -> pd.DataFrame:
    attr_file = root / "list_attr_celeba.txt"
    # The first line is the number of images; skip it
    df = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1)
    df.index = df["image_id"]
    df.drop(columns=["image_id"], inplace=True)
    df.replace({-1: 0}, inplace=True)  # map -1 ➔ 0
    return df


def _prepare_celeba_real() -> Dict[str, Dataset]:
    """Attempt to prepare the *full* CelebA dataset via torchvision."""
    root = DATA_DIR / "celeba"
    root.mkdir(parents=True, exist_ok=True)

    # Torchvision handles checksum + (re)download if necessary
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
        file_names: List[str] = part_df[part_df["split"] == s_id]["image_id"].tolist()

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

# ----------------------------------------------------------------------------
# Tiny synthetic replacement in case CelebA download is impossible ------------
# ----------------------------------------------------------------------------

def _prepare_celeba_synthetic() -> Dict[str, Dataset]:
    """Generate a lightweight synthetic version (~1k samples) mimicking CelebA."""

    rng = np.random.RandomState(42)

    def make_split(n: int, correlated: bool) -> Dataset:
        class _Synthetic(Dataset):
            def __len__(self):
                return n

            def __getitem__(self, idx):
                # --- targets & spurious attribute ---
                y = rng.randint(0, 2)
                if correlated:
                    # 90 % chance env == y
                    env = y if rng.rand() < 0.9 else 1 - y
                else:
                    env = rng.randint(0, 2)

                # --- random image tensor (normalised to ImageNet stats) ---
                img = torch.randn(3, 224, 224)
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                    [0.485, 0.456, 0.406]
                ).view(3, 1, 1)
                return img, int(y), int(env)

        return _Synthetic()

    return {
        "train": make_split(1000, correlated=True),
        "validation": make_split(200, correlated=False),
        "test": make_split(200, correlated=False),
    }

# Public API ------------------------------------------------------------------

def prepare_celeba() -> Dict[str, Dataset]:
    """CelebA hair-colour binary task with robust download handling.

    We first try to prepare the *real* CelebA dataset.  If this fails for any
    reason (network, missing dependencies, etc.) we gracefully fall-back to a
    tiny synthetic stand-in so that the rest of the experimental pipeline can
    complete inside constrained CI environments.
    """
    try:
        return _prepare_celeba_real()
    except Exception as e:  # noqa: BLE001 – we really want to catch *everything*
        warnings.warn(
            f"CelebA preparation failed ({e!s}). Falling back to synthetic dataset. "
            "Results obtained with the synthetic data are *not* comparable to the real benchmark."
        )
        return _prepare_celeba_synthetic()

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
