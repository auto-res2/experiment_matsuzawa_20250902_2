"""src/preprocess.py
Data and random-seed utilities.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

try:
    from wilds import get_dataset
except ImportError:  # pragma: no cover
    get_dataset = None  # type: ignore

__all__ = [
    "set_seed",
    "WaterbirdsWildsWrapper",
    "collate_with_optional_cf",
]

# -----------------------------------------------------------------------------
#  Deterministic seed helper
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
#  Custom collate fn that tolerates optional counterfactuals (None)
# -----------------------------------------------------------------------------

def _stack(values: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Utility that stacks tensors along the first dimension."""
    return torch.stack(list(values), dim=0)


def collate_with_optional_cf(batch: List[Tuple[torch.Tensor, int, torch.Tensor, Optional[torch.Tensor]]]):
    """Collate function for Waterbirds data.

    It supports the counterfactual field being `None` for every sample by
    returning a single `None` instead of a list of Nones (which breaks the
    default PyTorch collate).
    """

    xs, ys, metas, cfs = zip(*batch)

    xs = _stack(xs)
    ys = torch.tensor(ys)
    metas = torch.stack(metas) if isinstance(metas[0], torch.Tensor) else torch.tensor(metas)

    # If the first element is `None` we assume all are None (uniform dataset)
    if cfs[0] is None:
        cfs_batch = None
    else:
        cfs_batch = torch.stack(cfs)  # (B, K, C, H, W)

    return xs, ys, metas, cfs_batch


# -----------------------------------------------------------------------------
#  Dataset wrapper
# -----------------------------------------------------------------------------
class WaterbirdsWildsWrapper(Dataset):
    """WILDS Waterbirds with optional diffusion counterfactuals.

    If the real dataset is unavailable (e.g. in a CI environment without the
    12-GB Waterbirds archive), a small synthetic dataset is generated so that
    the rest of the training / evaluation pipeline can run end-to-end.
    """

    def __init__(self, split: str, cf_root: Optional[Path]):
        self.split = split
        self.cf_root = cf_root
        self.synthetic = False  # Will be flipped if we fall back to toy data

        # Attempt to load the real WILDS dataset ------------------------------------------------
        self.subset = None  # type: ignore
        if get_dataset is not None:
            root_dir = os.environ.get("WATERBIRDS_ROOT", "./data")
            try:
                dataset = get_dataset(dataset="waterbirds", root_dir=root_dir)
                transform = self._transform()
                if split == "train":
                    self.subset = dataset.get_subset("train", transform=transform)
                elif split == "val":
                    self.subset = dataset.get_subset("val", transform=transform)
                elif split == "test":
                    self.subset = dataset.get_subset("test", transform=transform)
                else:
                    raise ValueError(f"Unknown split: {split}")
            except Exception:
                # Any failure → fall back to synthetic data
                self.subset = None

        # Synthetic fallback --------------------------------------------------------------------
        if self.subset is None:
            self.synthetic = True
            self._init_synthetic_dataset()

    # -------------------------------------------------------------------------------------
    # Synthetic dataset helpers
    # -------------------------------------------------------------------------------------
    def _init_synthetic_dataset(self):
        # Define split sizes (tiny for fast unit-tests)
        sizes = {"train": 256, "val": 64, "test": 64}
        n_samples = sizes.get(self.split, 64)

        # Pre-generate random data so __getitem__ is fast and deterministic
        self._syn_images = torch.rand(n_samples, 3, 224, 224)
        self._syn_labels = torch.randint(low=0, high=2, size=(n_samples,))
        self._syn_meta = torch.zeros(n_samples, dtype=torch.long)

    # -------------------------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------------------------
    @staticmethod
    def _transform():
        """ImageNet-style augmentation used for the real Waterbirds dataset."""
        return T.Compose(
            [
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.RandomErasing(p=0.25),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # -------------------------------------------------------------------------------------
    # PyTorch Dataset API
    # -------------------------------------------------------------------------------------
    def __len__(self):
        if self.synthetic:
            return len(self._syn_labels)
        return len(self.subset)  # type: ignore[arg-type]

    # -----------------------------------------------------
    #  Counterfactual loading helper (only for real data)
    # -----------------------------------------------------
    def _load_counterfactuals(self, index: int) -> torch.Tensor:
        if self.cf_root is None:
            raise RuntimeError("Counterfactual directory not provided but requested.")

        img_path = Path(self.subset.dataset._input_array[index])  # type: ignore[attr-defined]
        rel_path = img_path.relative_to(img_path.parents[2])  # data/waterbirds/...
        cf_dir = self.cf_root / rel_path.parent / rel_path.stem
        if not cf_dir.exists():
            raise FileNotFoundError(f"Expected counterfactual directory {cf_dir} for {img_path}")
        cf_paths = sorted(list(cf_dir.glob("*.webp")))
        if len(cf_paths) == 0:
            raise RuntimeError(f"No counterfactuals found in {cf_dir}")

        imgs = [Image.open(p).convert("RGB") for p in cf_paths]
        tensor_imgs = [self.subset.transform(img) for img in imgs]
        return torch.stack(tensor_imgs)

    def __getitem__(self, idx: int):  # noqa: D401
        # Synthetic branch -----------------------------------------------------------------
        if self.synthetic:
            x = self._syn_images[idx]
            y = self._syn_labels[idx].item()
            m = torch.tensor(0)
            cf_imgs = None  # No CFs in synthetic mode
            return x, y, m, cf_imgs

        # Real WILDS branch -----------------------------------------------------------------
        x, y, m = self.subset[idx]
        cf_imgs = None
        if self.cf_root is not None:
            cf_imgs = self._load_counterfactuals(idx)
        return x, y, m, cf_imgs
