"""src/preprocess.py
Data-related utilities: downloading, extraction, dataset wrappers and
DataLoader builders.

Patch note (2025-09-02):
    • Fixed WaterbirdsDataset path resolver – images were looked up under a
      duplicated directory segment (…/waterbird_complete95_forest2water2/
      waterbird_complete95_forest2water2/…).  We now try both
      <base_dir>/<rel_path> and <images_root>/<rel_path> and pick the one
      that exists, preventing FileNotFoundError in DataLoader workers.
"""
from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path
from typing import Tuple

import requests
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import CelebA
from PIL import Image
import pandas as pd

# ---------------------------------------------------------------------
# Paths (resolved at import time so that other modules can use them)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# URLs for external datasets
# ---------------------------------------------------------------------
URLS = {
    "waterbirds": "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
}

# ---------------------------------------------------------------------
# Helper: SHA-256 for caching (not strictly used but handy)
# ---------------------------------------------------------------------

def hash_file(path: Path, algo: str = "sha256", chunk_size: int = 8192) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------
# Download + extraction helpers
# ---------------------------------------------------------------------

def download_url(url: str, dest: Path, desc: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[download_url] {desc} already at {dest}, skipping download.")
        return dest
    print(f"Downloading {desc} …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm.wrapattr(open(dest, "wb"), "write", miniters=1, total=total, desc=desc) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return dest


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract *tar(.gz/.tgz)* or *zip* archives.

    Previous logic compared suffix strings without the leading dot which
    caused the Waterbirds *.tar.gz* file to be opened as an un-compressed
    tar and therefore `tarfile` raised *ReadError: invalid header*.
    The check now correctly includes the dot and falls back to trying the
    alternative mode if the first attempt fails.
    """
    print(f"Extracting {archive_path} …")
    # If directory exists *and* is non-empty, assume extraction already done
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print("Archive already extracted, skipping.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ----------------  TAR archives  ----------------
        if (
            archive_path.suffixes[-2:] == [".tar", ".gz"]
            or archive_path.suffix in {".tar", ".tgz"}
        ):
            mode = "r:gz" if archive_path.suffixes[-1] == ".gz" or archive_path.suffix == ".tgz" else "r:"
            try:
                with tarfile.open(archive_path, mode) as tar:
                    tar.extractall(dest_dir)
            except tarfile.ReadError:
                # Fallback – try without gzip in case of mis-labelled file
                with tarfile.open(archive_path, "r:") as tar:
                    tar.extractall(dest_dir)
        # ----------------  ZIP archives  ----------------
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
    except Exception as exc:  # pragma: no cover – any extraction failure
        # Clean up to allow a fresh retry next run
        for p in dest_dir.glob("**/*"):
            p.unlink(missing_ok=True)
        dest_dir.rmdir()
        raise exc

# ---------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------

class WaterbirdsDataset(Dataset):
    """Waterbirds wrapper exposing image, label, group."""

    def __init__(self, root: Path, split: str, transform):
        assert split in {"train", "val", "test"}

        # ------------------------------------------------------------------
        # The Waterbirds archive extracts to the following hierarchy:
        #   <root>/
        #       waterbird_complete95_forest2water2/        <- base_dir
        #           metadata.csv
        #           waterbird_complete95_forest2water2/     <- images_root
        #               forest/
        #               water/
        # ------------------------------------------------------------------
        base_dir = root / "waterbird_complete95_forest2water2"
        meta_file = base_dir / "metadata.csv"
        images_root = base_dir / "waterbird_complete95_forest2water2"

        # In case users manually move files around, fall back to a recursive
        # search so that we do not crash hard but still warn.
        if not meta_file.exists():
            candidates = list(root.glob("**/metadata.csv"))
            if len(candidates) == 1:
                meta_file = candidates[0]
                images_root = meta_file.parent / "waterbird_complete95_forest2water2"
                print(f"[WaterbirdsDataset] Inferred metadata.csv at {meta_file}.")
            else:
                raise FileNotFoundError("metadata.csv missing – check extraction.")

        df = pd.read_csv(meta_file)
        split_map = {0: "train", 1: "val", 2: "test"}
        df = df[df.split.apply(lambda x: split_map[x] == split)].reset_index(drop=True)

        # ------------------------------------------------------------------
        # Resolve image paths robustly – handle cases where `img_filename`
        # already contains the inner directory segment (this previously led to
        # duplicated path components and FileNotFoundError).
        # ------------------------------------------------------------------
        def _resolve(rel_path: str) -> Path:
            p1 = images_root / rel_path  # most common case
            if p1.exists():
                return p1
            p2 = base_dir / rel_path     # fallback – rel already includes sub-dir
            if p2.exists():
                return p2
            # Final fallback: search – slower but ensures we do not crash in
            # rare edge cases where files are moved manually.
            matches = list(base_dir.glob(f"**/{rel_path.split('/')[-1]}"))
            if matches:
                return matches[0]
            raise FileNotFoundError(f"Image file not found for entry: {rel_path}")

        self.paths = [_resolve(p) for p in df["img_filename"].tolist()]
        self.y = df["y"].astype(int).values
        self.group = df["place"].astype(int).values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "x": img,
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "g": torch.tensor(self.group[idx], dtype=torch.long),
        }


class CelebAHairDataset(Dataset):
    """CelebA hair-colour dataset with spurious gender group."""

    def __init__(self, root: Path, split: str, transform):
        assert split in {"train", "val", "test"}
        self.dataset = CelebA(
            root=root,
            split="train" if split == "train" else "test",
            target_type="attr",
            download=True,
            transform=transform,
        )
        # Canonical 20k/20k val/test split as in prior work
        n_total = len(self.dataset)
        idxs = list(range(n_total))
        val_len = 20000
        test_len = 20000
        if split == "train":
            self.idxs = idxs[: n_total - val_len - test_len]
        elif split == "val":
            self.idxs = idxs[n_total - val_len - test_len : n_total - test_len]
        else:
            self.idxs = idxs[-test_len:]

        attr = self.dataset.attr[self.idxs]
        self.y = (attr[:, self.dataset.attr_names.index("Blond_Hair")] > 0).long()
        self.group = (attr[:, self.dataset.attr_names.index("Male")] > 0).long()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        img, _ = self.dataset[self.idxs[i]]
        return {
            "x": img,
            "y": self.y[i],
            "g": self.group[i],
        }

# ---------------------------------------------------------------------
# Transforms & DataLoaders (unchanged below)
# ---------------------------------------------------------------------
IMGNET_MEAN = (0.485, 0.456, 0.406)
IMGNET_STD = (0.229, 0.224, 0.225)

strong_aug = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.GaussianBlur(3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(IMGNET_MEAN, IMGNET_STD),
    ]
)


def build_dataloaders(dataset_name: str, batch_size: int):
    train_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMGNET_MEAN, IMGNET_STD),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMGNET_MEAN, IMGNET_STD),
        ]
    )

    if dataset_name == "waterbirds":
        archive = download_url(URLS["waterbirds"], DATA_ROOT / "waterbirds.tar.gz", "Waterbirds")
        extract_dir = DATA_ROOT / "waterbirds"
        extract_archive(archive, extract_dir)
        train_set = WaterbirdsDataset(extract_dir, "train", train_tf)
        val_set = WaterbirdsDataset(extract_dir, "val", val_tf)
        test_set = WaterbirdsDataset(extract_dir, "test", val_tf)
    elif dataset_name == "celeba":
        celeba_root = DATA_ROOT / "celeba"
        train_set = CelebAHairDataset(celeba_root, "train", train_tf)
        val_set = CelebAHairDataset(celeba_root, "val", val_tf)
        test_set = CelebAHairDataset(celeba_root, "test", val_tf)
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader, test_loader
