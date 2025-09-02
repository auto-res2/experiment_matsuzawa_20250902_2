"""
preprocess.py
Data loading / preprocessing logic for Waterbirds with counterfactuals.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

try:
    from wilds import get_dataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The `wilds` library is mandatory – install with `pip install wilds`."
    ) from e

# -----------------------------------------------------------------------------
# Image & transform config
# -----------------------------------------------------------------------------
IMG_SIZE = 224


class WaterbirdsWithCF(Dataset):
    """Waterbirds (WILDS) dataset that additionally provides diffusion CFs.

    Each __getitem__ returns:
        img, label, metadata, cf_imgs (Tensor[K,3,224,224] or None)
    """

    def __init__(self, split: str, cf_root: Optional[Path]):
        assert split in {"train", "val", "test"}, "Invalid split"
        self.wilds_data = get_dataset("waterbirds", root_dir=os.environ["WATERBIRDS_ROOT"])
        self.subset = self.wilds_data.get_subset(split, transform=self._build_transform(split))
        self.cf_root = Path(cf_root) if cf_root is not None else None
        self.split = split

    # ---------------------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------------------
    @staticmethod
    def _build_transform(split: str):
        train_t = split == "train"
        aug = [T.Resize(256)]
        if train_t:
            aug += [
                T.RandomResizedCrop(IMG_SIZE),
                T.RandomHorizontalFlip(),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.RandomErasing(p=0.25),
            ]
        else:
            aug.append(T.CenterCrop(IMG_SIZE))
        aug += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        return T.Compose(aug)

    # ------------------------------------------------------------------
    # Counterfactual loader
    # ------------------------------------------------------------------
    def _load_cf(self, original_rel_path: Path) -> Optional[torch.Tensor]:
        if self.cf_root is None:
            return None

        # CFs stored under <cf_root>/<same_rel_dir>/<imgStem>/cf_*.webp
        cf_dir = self.cf_root / original_rel_path.parent / original_rel_path.stem
        if not cf_dir.exists():
            raise FileNotFoundError(f"Counterfactual directory missing: {cf_dir}")
        cf_files = sorted(list(cf_dir.glob("*.webp")))
        if len(cf_files) == 0:
            raise RuntimeError(f"No counterfactual .webp files in {cf_dir}")

        imgs = []
        for p in cf_files:
            with Image.open(p).convert("RGB") as img_pil:
                imgs.append(self.subset.transform(img_pil))
        return torch.stack(imgs)  # (K,3,224,224)

    # ------------------------------------------------------------------
    # Standard Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):  # type: ignore[override]
        return len(self.subset)

    def __getitem__(self, idx):  # type: ignore[override]
        x, y, m = self.subset[idx]
        # relative path to find corresponding CF directory – assumes WILDS layout
        img_path = Path(self.subset.dataset._input_array[idx])  # type: ignore[attr-defined]
        rel = img_path.relative_to(img_path.parents[2])  # waterbird/... (2 levels up)
        cf_imgs = self._load_cf(rel) if self.cf_root is not None else None
        return x, int(y), m, cf_imgs
