from __future__ import annotations
import os
import random
import zipfile
from typing import Tuple

import numpy as np
import requests
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

#############################   1.  Reproducibility   #########################

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#############################   2.  External files   ##########################

URLS = {
    "cora_npz": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/cora.npz",
    "citeseer_npz": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/citeseer.npz",
    "pubmed_npz": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/pubmed.npz",
    "ogbn_arxiv": "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
}


def _download(url: str, out_path: str):
    if os.path.exists(out_path):
        return
    print(f"Downloading {url} → {out_path}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def prepare_external_files(root: str):
    os.makedirs(root, exist_ok=True)
    for _, url in URLS.items():
        fname = os.path.join(root, os.path.basename(url))
        _download(url, fname)
        if fname.endswith(".zip") and not os.path.exists(fname[:-4]):
            with zipfile.ZipFile(fname) as zf:
                zf.extractall(root)

#############################   3.  Data Loading   ############################

def load_planetoid_dataset(name: str, root: str, device: str) -> Tuple[Data, Planetoid]:
    dataset = Planetoid(root=root, name=name)
    data: Data = dataset[0]
    # ensure self-loops for stability
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    row, _ = data.edge_index
    deg = torch.bincount(row, minlength=data.num_nodes).float()
    data.deg_norm = (1.0 / deg.clamp(min=1.0)).to(torch.float32)

    return data.to(device), dataset
