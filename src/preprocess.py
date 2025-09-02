"""src/preprocess.py
Data loading, random-seed control and directory set-up.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.datasets import Planetoid, WebKB
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

def load_dataset(name: str):
    """Return a PyG data object pinned to the GPU.

    Only the datasets required by the quick sanity check are implemented here.
    """
    name = name.lower()
    try:
        if name in {"cora", "citeseer", "pubmed"}:
            ds = Planetoid(str(DATA_DIR), name.capitalize())
            data = ds[0].to(DEVICE)
        elif name in {"cornell", "texas", "wisconsin"}:
            ds = WebKB(str(DATA_DIR), name.capitalize())
            data = ds[0].to(DEVICE)
        else:
            raise ValueError(f"Dataset {name} not supported in quick mode.")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {name}: {e}") from e

    # Ensure self-loops are present (PairNorm not used here, so always add)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    return data
