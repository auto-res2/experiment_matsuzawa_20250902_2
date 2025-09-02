"""src/preprocess.py
---------------------------------------------------------------------
Data-handling utilities & project-wide constants live here.  Other
modules import (PROJECT_ROOT / DATA_ROOT / …) from this file to avoid
circular dependencies.
"""

from __future__ import annotations

# --------------------------- std-lib ------------------------------
import hashlib
import pathlib
from typing import Callable

# -------------------------- third-party ---------------------------
import requests
from tqdm import tqdm
import torch

# ==================================================================
#  Global constants & directory set-up
# ==================================================================

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data"
FIG_ROOT = PROJECT_ROOT / "figures"
LOG_ROOT = PROJECT_ROOT / "logs"

for _d in (DATA_ROOT, FIG_ROOT, LOG_ROOT):
    _d.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "FIG_ROOT",
    "LOG_ROOT",
    "DEVICE",
    "DTYPE",
    "download_and_extract",
]

# ==================================================================
#  Robust download + extract helper (fail-fast & checksum aware)
# ==================================================================

def _hash_file(path: pathlib.Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MiB chunks
            h.update(chunk)
    return h.hexdigest()


def download_and_extract(url: str, dest_dir: pathlib.Path, sha256: str | None = None):
    """Download *url* into *dest_dir* and extract if it is an archive.

    If the file already exists (and optionally matches the provided
    SHA-256 checksum) the download is skipped.  Extraction is only
    performed once and guarded via a *.extracted* flag file.
    """

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    archive_path = dest_dir / filename

    # ---------------------- download (if needed) -------------------
    if not archive_path.exists():
        print(f"Downloading {url} …")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(archive_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
    else:
        print(f"Found cached archive: {archive_path}")

    # ---------------------- checksum (optional) --------------------
    if sha256 is not None:
        digest = _hash_file(archive_path)
        if digest != sha256:
            raise RuntimeError(
                f"Checksum mismatch for {archive_path}: {digest} != {sha256}"
            )

    # ---------------------- extraction -----------------------------
    target_flag_file = dest_dir / (filename + ".extracted")
    if target_flag_file.exists():
        print(f"Archive already extracted: {archive_path}")
        return

    import tarfile, zipfile

    print(f"Extracting {archive_path} …")
    try:
        if filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=dest_dir)
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(path=dest_dir)
        else:
            raise RuntimeError(f"Unrecognised archive format: {filename}")
    finally:
        target_flag_file.touch()
