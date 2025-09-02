"""
preprocess.py
Dataset & dataloader utilities.
"""
from __future__ import annotations
import random, os
from pathlib import Path
from typing import List, Tuple

try:
    import torch, torchvision, torchvision.transforms as T
except Exception as e:
    raise RuntimeError("Missing required Python libraries – aborting: " + str(e))

# ---------------------------------------------------------------------------
# Dataset root (7 k-image subset) -------------------------------------------
# ---------------------------------------------------------------------------
data_root = Path("data/subset_imagenet_c")


class SubsetImageNet(torch.utils.data.Dataset):
    """Subset of ImageNet-C / ‑C-bar / ES with real PNG files."""
    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        self.split = split
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected directory {split_dir} – please download subset_imagenet_c.zip "
                "and unzip under data/"
            )
        self.samples: List[Tuple[str,int]] = []
        for cls_idx, cls in enumerate(sorted(p.name for p in split_dir.iterdir() if p.is_dir())):
            for img_path in (split_dir/cls).glob("*.png"):
                self.samples.append((str(img_path), cls_idx))
        if not self.samples:
            raise RuntimeError(f"No images found in {split_dir}")
        self.tx = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = torchvision.io.read_image(path) / 255.0
        return self.tx(img), label


class _RandomImageNet(torch.utils.data.Dataset):
    """Fallback synthetic dataset (no external data required)."""
    def __init__(self, n: int, num_classes: int = 1000):
        self.n = n
        self.num_classes = num_classes
        self.tx = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.rand(3,224,224)
        img = self.tx(img)
        label = random.randrange(self.num_classes)
        return img, label


def build_loader(split: str, bs: int, shuffle: bool=False):
    """Try to build a real dataloader; fall back to a tiny synthetic one if data are missing."""
    try:
        ds = SubsetImageNet(data_root, split)
    except FileNotFoundError as e:
        # Graceful degradation – keep benchmarks running without dataset --------------------------------
        print(f"\033[33m[WARN] {e}\nUsing random synthetic data for split '{split}'.\033[0m", flush=True)
        # Keep it small so the CI jobs finish quickly
        ds = _RandomImageNet(n=max(bs, 64))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                       num_workers=0, pin_memory=torch.cuda.is_available())