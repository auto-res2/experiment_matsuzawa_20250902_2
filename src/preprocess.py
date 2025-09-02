"""src/preprocess.py
Data loading, random-seed control and directory set-up.
Fixed: WikipediaNetwork provides boolean masks with shape (N, #splits).
We now collapse those masks to the first split so that downstream code
(which expects 1-D boolean vectors) works unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.utils import add_self_loops

# -----------------------------------------------------------------------------
#  Project directories
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
for _d in [DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda")

# -----------------------------------------------------------------------------
#  Reproducibility helper
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    """Seed Python RNGs, NumPy and PyTorch (including CUDA)."""
    import random, numpy as np  # local import keeps global namespace clean

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
#  Dataset loader (quick-mode subset)
# -----------------------------------------------------------------------------

def _load_planetoid(name: str):
    ds = Planetoid(str(DATA_DIR), name.capitalize())
    return ds[0]


def _load_webkb(name: str):
    ds = WebKB(str(DATA_DIR), name.capitalize())
    return ds[0]


def _load_wikipedia(name: str):
    # WikipediaNetwork supports lowercase names
    ds = WikipediaNetwork(str(DATA_DIR), name.lower(), geom_gcn_preprocess=True)
    return ds[0]


# -----------------------------------------------------------------------------
#  Mask sanitiser – collapses 2-D boolean masks to 1-D (first split)
# -----------------------------------------------------------------------------

def _ensure_1d_masks(data):
    """Many heterophilous datasets ship with 10 different train/val/test splits
    stored as boolean matrices of shape (N, 10).  The rest of the pipeline
    expects 1-D masks.  We therefore squeeze to the first split if needed."""

    for attr in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data, attr):
            m = getattr(data, attr)
            # We assume masks are boolean or uint8.  If 2-D, take first column.
            if m.dim() == 2:
                setattr(data, attr, m[:, 0])
    return data


# -----------------------------------------------------------------------------
#  Public loader
# -----------------------------------------------------------------------------

def load_dataset(name: str):
    """Return a PyG data object pinned to the GPU.
    Only the datasets required by the quick sanity check are implemented here.
    """
    name = name.lower()
    try:
        if name in {"cora", "citeseer", "pubmed"}:
            data = _load_planetoid(name)
        elif name in {"cornell", "texas", "wisconsin"}:
            data = _load_webkb(name)
        elif name in {"chameleon", "squirrel", "crocodile"}:
            data = _load_wikipedia(name)
        else:
            raise ValueError(f"Dataset {name} not supported in quick mode.")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {name}: {e}") from e

    # Ensure masks are 1-D (first split)
    data = _ensure_1d_masks(data)

    # Ensure self-loops are present (PairNorm not used here, so always add)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    return data.to(DEVICE)
