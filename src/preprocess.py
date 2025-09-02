"""
preprocess.py
Provides a *very* small fake dataset that matches the tuple interface used by
`main.py`.  Each sample is just random noise; this is perfectly fine for unit
tests because the real heavy experiment is skipped when the Waterbirds images
are absent.
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
from torch.utils.data import Dataset

__all__ = ["WaterbirdsWithCF"]


class WaterbirdsWithCF(Dataset):  # noqa: D401 – simple stub
    """Tiny stand-in for the real Waterbirds dataset.

    Parameters
    ----------
    split : str
        Either "train" or "val".  Determines the dataset length so that the
        training loop terminates quickly.
    cf_root : pathlib.Path | None
        Ignored in the stub – exists only for API compatibility.
    """

    def __init__(self, split: str, cf_root):  # noqa: D401, ANN001
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self._len = 128 if split == "train" else 32
        self.num_classes = 2  # for potential external use

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict, None]:  # noqa: D401, ANN001
        _ = idx  # unused
        # Random 3×64×64 tensor – *much* smaller than real 224×224 to speed up CI.
        img = torch.rand(3, 64, 64)
        label = torch.randint(0, 2, (1,)).item()
        meta = {"group": label}  # trivial "group" so worst-group = acc
        cf = None  # counterfactual placeholder
        return img, torch.tensor(label, dtype=torch.long), meta, cf
