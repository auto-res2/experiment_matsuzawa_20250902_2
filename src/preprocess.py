"""src/preprocess.py
Utility functions shared across the project: reproducibility helpers, file/dir
management, data download / extraction, device settings …
"""
from __future__ import annotations
import os, random, tarfile, zipfile
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
import requests

# -----------------------------------------------------------------------------
#  Paths & global flags
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
LOG_DIR = os.path.join(ROOT, "logs")
for _d in (DATA_DIR, FIG_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUICK_MODE = os.getenv("FULL_RUN", "0") != "1"  # default quick for CI

# -----------------------------------------------------------------------------
#  Reproducibility & download helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_archive(path: str, out_dir: str) -> None:
    """Extract .zip / .tar.gz / .tgz archives to *out_dir*"""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
    elif path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(out_dir)


def download(url: str, out_path: str) -> None:
    """Download *url* to *out_path* unless the file already exists. Archive types
    (.zip / .tar.gz / .tgz) are extracted automatically into *DATA_DIR*.
    """
    if os.path.exists(out_path):
        return
    print(f"Downloading {url} → {out_path}")
    try:
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(f"Failed to download {url}: {e}") from e

    # Auto-extract
    if out_path.endswith((".zip", ".tar.gz", ".tgz")):
        _extract_archive(out_path, DATA_DIR)
