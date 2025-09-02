"""
preprocess.py – data download / loading / dataset utilities
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)
CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Helper – SHA-256 checksum
# -----------------------------------------------------------------------------

def _sha256sum(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -----------------------------------------------------------------------------
# Download / extract utilities
# -----------------------------------------------------------------------------

def download_file(url: str, target: Path, expected_sha256: str | None = None) -> None:
    """Download to *target* with optional checksum validation."""
    if target.exists() and (
        expected_sha256 is None or _sha256sum(target) == expected_sha256
    ):
        print(f"✓ {target.name} already downloaded.")
        return

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {target.name}") as pbar:
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    if expected_sha256:
        assert _sha256sum(target) == expected_sha256, "Checksum mismatch – file may be corrupted."


def extract_tar(src: Path, dest: Path):
    import tarfile

    assert src.exists(), f"Archive {src} not found."
    if dest.exists():
        print(f"✓ {dest} already extracted.")
        return
    print(f"Extracting {src.name} …")
    with tarfile.open(src) as tf:
        tf.extractall(dest)

# -----------------------------------------------------------------------------
# Waterbirds dataset (WILDS)
# -----------------------------------------------------------------------------
WATERBIRDS_URL = "https://storage.googleapis.com/wilds-datasets/waterbirds_v1.1.tar.gz"
WATERBIRDS_SHA256 = "21808c3b44c7fc5e7d86c3d850a4e6ad0bca5e2837fae3fe9ff8261cd17997e3"


def prepare_waterbirds() -> Path:
    dest = DATA_ROOT / "waterbirds"
    if dest.exists():
        return dest
    archive = CACHE_ROOT / "waterbirds.tar.gz"
    download_file(WATERBIRDS_URL, archive, WATERBIRDS_SHA256)
    extract_tar(archive, DATA_ROOT)
    assert dest.exists(), "Extraction failed – directory missing."
    return dest

# -----------------------------------------------------------------------------
# CelebA dataset path utility (torchvision handles the download internally)
# -----------------------------------------------------------------------------

def prepare_celeba() -> Tuple[Path, Path]:
    celeba_root = DATA_ROOT / "celeba"
    celeba_root.mkdir(exist_ok=True)
    return celeba_root, celeba_root

# -----------------------------------------------------------------------------
# Synthetic Perlin noise (used for Syn-Freq-Waterbirds creation; optional)
# -----------------------------------------------------------------------------

def perlin_noise(size: int = 128, scale: int = 8) -> np.ndarray:
    """Generate 2-D Perlin noise – utility for synthetic dataset construction."""

    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = scale / size
    d = size // scale
    gradients = np.random.randn(d, d, 2)
    gradients /= np.linalg.norm(gradients, axis=-1, keepdims=True)

    def tile(a):
        return np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)

    g00 = tile(gradients)
    g10 = np.roll(g00, shift=-scale, axis=1)
    g01 = np.roll(g00, shift=-scale, axis=0)
    g11 = np.roll(g10, shift=-scale, axis=0)

    xs = np.linspace(0, 1, scale, endpoint=False)
    ys = xs[:, None]
    wx = f(xs)
    wy = f(ys)

    noise = np.zeros((size, size))
    for i in range(d):
        for j in range(d):
            ix, iy = i * scale, j * scale
            dot00 = (g00[ix, iy] * np.dstack((xs, ys))).sum(-1)
            dot10 = (g10[ix, iy] * np.dstack((xs - 1, ys))).sum(-1)
            dot01 = (g01[ix, iy] * np.dstack((xs, ys - 1))).sum(-1)
            dot11 = (g11[ix, iy] * np.dstack((xs - 1, ys - 1))).sum(-1)
            nx0 = dot00 * (1 - wx) + dot10 * wx
            nx1 = dot01 * (1 - wx) + dot11 * wx
            nxy = nx0 * (1 - wy) + nx1 * wy
            noise[i * scale : (i + 1) * scale, j * scale : (j + 1) * scale] = nxy
    return (noise - noise.min()) / (noise.max() - noise.min())

# -----------------------------------------------------------------------------
# Waterbirds DataLoaders (train / val / test)
# -----------------------------------------------------------------------------

def waterbirds_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = prepare_waterbirds()
    metadata_path = root / "metadata.csv"
    images_folder = root / "images"
    assert metadata_path.exists(), "Waterbirds metadata.csv missing."

    meta = pd.read_csv(metadata_path)
    img_paths = meta["img_filename"].tolist()
    labels = meta["y"].tolist()

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
        ]
    )
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ------------------------------------------------------------------
    # Dataset wrapper
    # ------------------------------------------------------------------
    class _Waterbirds(torch.utils.data.Dataset):
        def __init__(self, indices: List[int], train: bool):
            self.indices = indices
            self.train = train
            self.transform = transform_train if train else transform_eval

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img_path = images_folder / img_paths[real_idx]
            y = labels[real_idx]
            img = torchvision.io.read_image(str(img_path)).float() / 255.0
            img = self.transform(transforms.ToPILImage()(img))
            return img, y

    # deterministic split (WILDS meta provides split column 0/1/2)
    train_idx = [i for i, s in enumerate(meta["split"].tolist()) if s == 0]
    val_idx = [i for i, s in enumerate(meta["split"].tolist()) if s == 1]
    test_idx = [i for i, s in enumerate(meta["split"].tolist()) if s == 2]

    loaders = []
    for idxs, is_train in zip([train_idx, val_idx, test_idx], [True, False, False]):
        ds = _Waterbirds(idxs, is_train)
        loaders.append(
            DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True)
        )

    return tuple(loaders)
