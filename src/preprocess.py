"""src/preprocess.py
Data and random-seed utilities.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

try:
    from wilds import get_dataset
except ImportError:  # pragma: no cover
    get_dataset = None  # type: ignore

__all__ = ["set_seed", "WaterbirdsWildsWrapper"]


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
#  Dataset wrapper
# -----------------------------------------------------------------------------
class WaterbirdsWildsWrapper(Dataset):
    """WILDS Waterbirds with optional diffusion counterfactuals."""

    def __init__(self, split: str, cf_root: Optional[Path]):
        if get_dataset is None:
            raise ImportError(
                "wilds library is required but not installed. Install via `pip install wilds`."
            )

        self.dataset = get_dataset(
            dataset="waterbirds", root_dir=os.environ.get("WATERBIRDS_ROOT", "./data")
        )
        self.split = split
        self.cf_root = cf_root

        transform = self._transform()
        if split == "train":
            self.subset = self.dataset.get_subset("train", transform=transform)
        elif split == "val":
            self.subset = self.dataset.get_subset("val", transform=transform)
        elif split == "test":
            self.subset = self.dataset.get_subset("test", transform=transform)
        else:
            raise ValueError(f"Unknown split: {split}")

    # ---------------------
    # Static helpers
    # ---------------------
    @staticmethod
    def _transform():
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

    # ---------------------
    # PyTorch Dataset API
    # ---------------------
    def __len__(self):
        return len(self.subset)

    # -----------------------------------------------------
    #  Counterfactual loading helper (optional at runtime)
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
        x, y, m = self.subset[idx]
        cf_imgs = None
        if self.cf_root is not None:
            cf_imgs = self._load_counterfactuals(idx)
        return x, y, m, cf_imgs
