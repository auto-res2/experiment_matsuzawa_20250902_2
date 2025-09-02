"""src/preprocess.py – data downloading / preprocessing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from torch_geometric.utils import add_self_loops

# -----------------------------------------------------------------------------
# Directories ------------------------------------------------------------------
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset loader ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_dataset(name: str) -> Any:
    """Load a small-/medium-scale benchmark dataset and apply basic transforms."""

    name_l = name.lower()
    try:
        if name_l in {"cora", "citeseer", "pubmed"}:
            ds = Planetoid(str(DATA_DIR), name_l.capitalize())
            data = ds[0]
        elif name_l in {"chameleon", "squirrel"}:
            ds = WikipediaNetwork(str(DATA_DIR), name_l, geom_gcn_preprocess=False)
            data = ds[0]
        elif name_l in {"cornell", "texas", "wisconsin"}:
            ds = WebKB(str(DATA_DIR), name_l.capitalize())
            data = ds[0]
        else:
            raise ValueError(f"Unknown dataset '{name}'.")
    except Exception as e:  # pragma: no cover – I/O related
        raise RuntimeError(f"Failed to download / load dataset '{name}': {e}") from e

    # add self-loops & L2 normalise features ---------------------------------
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    data.x = F.normalize(data.x, p=2, dim=-1)
    return data


__all__ = ["load_dataset", "DATA_DIR"]
