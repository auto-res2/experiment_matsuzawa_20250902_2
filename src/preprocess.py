"""
preprocess.py – data loading, transforms, determinism helpers
"""
from __future__ import annotations
import random, os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset

# -------------------------------------------------------------------------------------
#  Repository root & folders ----------------------------------------------------------
# -------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data";    DATA_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------------------------
#  Determinism helpers ----------------------------------------------------------------
# -------------------------------------------------------------------------------------


def _enable_determinism() -> None:
    """Enable (best-effort) deterministic behaviour without raising errors.

    Some CUDA operations – notably GEMMs executed via cuBLAS – have no fully
    deterministic implementation.  When ``torch.use_deterministic_algorithms`` is
    set *strictly* (``warn_only=False``), PyTorch raises a ``RuntimeError`` the
    first time such an op is encountered.  This repository only requires
    *reproducibility* rather than *bit-exact* deterministic results, so we opt for
    a pragmatic compromise:

    1.  The necessary cuBLAS workspace configuration variable is exported **before
        any GPU kernels are launched**.  This selects an alternative algorithm
        that is deterministic up to the limits of floating-point arithmetic.
    2.  ``torch.use_deterministic_algorithms`` is called with ``warn_only=True`` so
        that PyTorch logs a warning instead of aborting when it cannot guarantee
        strict determinism.
    """
    # (1) cuBLAS reproducibility -----------------------------------------------------
    # Needs to be set *before* the first CUDA context is initialised.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    # (2) Activate PyTorch deterministic mode but do *not* raise on violations.
    # The ``warn_only`` keyword is available from PyTorch ≥1.11.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Fallback for very old PyTorch versions that do not expose warn_only.
        # In that case we prefer turning determinism *off* to avoid runtime
        # crashes that would otherwise occur during the backward pass.
        torch.use_deterministic_algorithms(False)


def set_seed(seed: int):
    """Seed all RNGs and switch PyTorch to (best-effort) deterministic behaviour."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    _enable_determinism()

# -------------------------------------------------------------------------------------
#  Image transforms -------------------------------------------------------------------
# -------------------------------------------------------------------------------------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

TEST_TF = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

# -------------------------------------------------------------------------------------
#  Continual-learning task splits ------------------------------------------------------
# -------------------------------------------------------------------------------------

def _split_by_class(ds: datasets.VisionDataset, order: List[int], classes_per_task: int) -> List[Subset]:
    """Return a list of Subsets, each with *classes_per_task* contiguous classes."""
    targets = np.array(ds.targets)
    tasks = []
    for i in range(0, len(order), classes_per_task):
        cls = order[i : i + classes_per_task]
        idx = np.where(np.isin(targets, cls))[0]
        tasks.append(Subset(ds, idx))
    return tasks


def get_split_cifar(order: List[int] | None = None) -> Tuple[List[Subset], List[Subset]]:
    if order is None:
        order = list(range(100))
        random.shuffle(order)
    tr = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=TRAIN_TF)
    te = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=TEST_TF)
    return _split_by_class(tr, order, 10), _split_by_class(te, order, 10)