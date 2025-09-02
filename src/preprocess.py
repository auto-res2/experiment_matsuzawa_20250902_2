"""src/preprocess.py – dataset utilities, download helpers, reproducibility seed"""
import random
import tarfile
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import requests
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Reproducibility -------------------------------------------------------------
# -----------------------------------------------------------------------------
SEEDS = [11, 17, 23, 29, 31]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Download helper -------------------------------------------------------------
# -----------------------------------------------------------------------------
CHUNK = 2 ** 20  # 1 MB

def _http_download(url: str, dst: Path):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dst, "wb") as fp, tqdm(total=total, unit="B", unit_scale=True, desc=f"HTTP {url}") as bar:
        for chunk in r.iter_content(CHUNK):
            if chunk:
                fp.write(chunk)
                bar.update(len(chunk))


def _gdrive_download(url: str, dst: Path):
    import gdown

    gdown.download(url, str(dst), quiet=False, fuzzy=True)


def download_and_extract(url: str, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    tmp = extract_to / "_tmp_download"

    if url.startswith("http") and "drive.google.com" not in url:
        _http_download(url, tmp)
    else:
        _gdrive_download(url, tmp)

    # ------------ extract if archive -----------------------------------------
    if tarfile.is_tarfile(tmp):
        with tarfile.open(tmp) as tar:
            tar.extractall(path=extract_to)
        tmp.unlink()
    elif zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(path=extract_to)
        tmp.unlink()
    else:  # raw file
        (extract_to / url.split("/")[-1]).write_bytes(tmp.read_bytes())

# -----------------------------------------------------------------------------
# Datasets --------------------------------------------------------------------
# -----------------------------------------------------------------------------
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


class SplitCIFAR100:
    """20 × 5-class Split CIFAR-100 continual benchmark."""

    IMG_SIZE = 32

    def __init__(self, root: Path, seed: int):
        self.root = root
        self.seed = seed
        self._make_tasks()

    # ---------------------------------------------------------------------
    def _make_tasks(self):
        full_train = datasets.CIFAR100(self.root, train=True, download=True)
        full_test = datasets.CIFAR100(self.root, train=False, download=True)
        cls_order = torch.randperm(100, generator=torch.Generator().manual_seed(self.seed))
        self.tasks = [cls_order[i : i + 5].tolist() for i in range(0, 100, 5)]
        self.train_ds, self.test_ds = full_train, full_test

    # ---------------------------------------------------------------------
    def _split_subset(self, indices: List[int]):
        random.shuffle(indices)
        n = len(indices)
        n_val = n // 20
        n_test = n // 10
        return indices[n_val + n_test :], indices[: n_val], indices[n_val : n_val + n_test]

    @staticmethod
    def _build_transforms():
        aug = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.2, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        return aug, val

    # ---------------------------------------------------------------------
    def get_task(self, idx: int):
        classes = self.tasks[idx]
        aug, val = self._build_transforms()

        tr_idx = [i for i, y in enumerate(self.train_ds.targets) if y in classes]
        te_idx = [i for i, y in enumerate(self.test_ds.targets) if y in classes]
        tr_idx, val_idx, te_idx = self._split_subset(tr_idx)

        self.train_ds.transform = aug
        self.test_ds.transform = val

        train = torch.utils.data.Subset(self.train_ds, tr_idx)
        val_ds = torch.utils.data.Subset(self.train_ds, val_idx)
        test = torch.utils.data.Subset(self.test_ds, te_idx)
        return train, val_ds, test, classes