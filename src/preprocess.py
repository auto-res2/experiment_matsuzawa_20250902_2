"""src/preprocess.py
Data utilities: downloading helpers and dataset wrappers.
"""
from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path
from typing import Tuple, List, Dict

import requests
from tqdm import tqdm
import gdown

import torch
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Global data directory – automatically created on first import
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 1024 * 1024  # 1 MB


# -----------------------------------------------------------------------------
# Generic download & extract helpers
# -----------------------------------------------------------------------------

def _sha256(fname: Path) -> str:
    h = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_http(url: str, dst: Path) -> None:
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {url}") as pbar:
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _download_gdrive(url: str, dst: Path) -> None:
    gdown.download(url, str(dst), quiet=False, fuzzy=True)


def download_and_extract(url: str, extract_to: Path) -> None:
    """Download a (possibly archive) file from *url* and extract into *extract_to*."""
    extract_to.mkdir(parents=True, exist_ok=True)
    tmp = extract_to / "tmp_download"

    if url.startswith("http") and "drive.google.com" not in url:
        _download_http(url, tmp)
    else:
        _download_gdrive(url, tmp)

    # ---------------- Archive handling ----------------
    try:
        if tarfile.is_tarfile(tmp):
            with tarfile.open(tmp) as tar:
                tar.extractall(extract_to)
        elif zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as zf:
                zf.extractall(extract_to)
        else:
            dst = extract_to / Path(url).name
            tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Helper: dataset wrapper that remaps class labels to a contiguous range
# -----------------------------------------------------------------------------
class _RemapDataset(torch.utils.data.Dataset):
    """Wrap a (subset of a) dataset and remap labels using *label_map* (orig→new)."""

    def __init__(self, subset: torch.utils.data.Subset, label_map: Dict[int, int]):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.label_map[int(y)]


# -----------------------------------------------------------------------------
# Continual-Learning data stream – CIFAR-100 split into 20×5 classes
# -----------------------------------------------------------------------------
class SplitCIFAR100:
    """Utility that returns task-specific datasets for the 20×5-class CIFAR-100 stream."""

    IMG_SIZE = 32

    def __init__(self, root: Path, seed: int) -> None:
        self.root = root
        self.seed = seed
        self.tasks_order, self.full_train, self.full_test = self._build_tasks()

    # ------------------------------------------------------------------
    def _build_tasks(self):
        full_train = datasets.CIFAR100(self.root, train=True, download=True)
        full_test = datasets.CIFAR100(self.root, train=False, download=True)
        rng = torch.Generator().manual_seed(self.seed)
        classes = torch.randperm(100, generator=rng).tolist()
        tasks: List[List[int]] = []
        for i in range(0, 100, 5):
            tasks.append(classes[i : i + 5])
        return tasks, full_train, full_test

    # ------------------------------------------------------------------
    def get_task_datasets(self, task_id: int):
        tasks, tr_all, te_all = self.tasks_order, self.full_train, self.full_test
        cls = tasks[task_id]

        # Build orig→new label map (e.g. 17→0, 2→1...)
        label_map = {c: i for i, c in enumerate(cls)}

        tr_idx = [i for i, y in enumerate(tr_all.targets) if y in cls]
        te_idx = [i for i, y in enumerate(te_all.targets) if y in cls]

        prep_train = transforms.Compose(
            [
                transforms.RandomCrop(self.IMG_SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.2, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276)),
            ]
        )
        prep_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276)),
            ]
        )

        train_subset = torch.utils.data.Subset(tr_all, tr_idx)
        train_subset.dataset.transform = prep_train
        test_subset = torch.utils.data.Subset(te_all, te_idx)
        test_subset.dataset.transform = prep_test

        # Wrap with remap wrapper to ensure labels are 0..len(cls)-1
        train_ds = _RemapDataset(train_subset, label_map)
        test_ds = _RemapDataset(test_subset, label_map)

        return train_ds, test_ds, cls
