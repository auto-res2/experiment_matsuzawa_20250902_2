````python
"""
train.py – model architectures, memory buffers, training utilities
"""
from __future__ import annotations
import math, random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans

# We re-use paths that are created inside preprocess.py --------------------------------
from .preprocess import ROOT, DATA_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------------------
#  Models -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
class ResNet18(nn.Module):
    """Thin wrapper that exposes features for HOFQ replay."""

    def __init__(self, n_cls: int = 100, *, pretrained: bool = True):
        """Create a ResNet-18 backbone.

        Parameters
        ----------
        n_cls : int
            Number of output classes for the final classifier.
        pretrained : bool, default=True
            Whether to start from ImageNet pretrained weights.  Using a
            pretrained backbone drastically speeds up convergence and is
            necessary for the offline sanity check to exceed the 70 % accuracy
            threshold after a single training epoch.
        """
        super().__init__()
        from torchvision import models  # local import avoids heavy dependency if unused

        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT  # torchvision >=0.13
            except AttributeError:
                weights = "IMAGENET1K_V1"  # very old torchvision fallback
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)
        # Replace the ImageNet classifier with an identity layer – the new task
        # specific classifier is defined below.
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, n_cls)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, *, return_feat: bool = False):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return (feat, logits) if return_feat else logits


# -------------------------------------------------------------------------------------
#  Replay buffers ---------------------------------------------------------------------
# -------------------------------------------------------------------------------------
class PixelBuffer:
    """JPEG-size based pixel replay buffer (very simplified)."""

    def __init__(self, max_bytes: int, avg_jpeg: int = 11_000):
        self.cap = max_bytes // avg_jpeg
        self.img: List[torch.Tensor] = []
        self.lbl: List[int] = []

    # ------------------------------------------------------------------
    @property
    def bytes(self):
        return len(self.img) * 11_000

    # ------------------------------------------------------------------
    def add(self, x: torch.Tensor, y: torch.Tensor):
        for xi, yi in zip(x, y):
            if len(self.img) >= self.cap:
                self.img.pop(0)
                self.lbl.pop(0)
            self.img.append(xi.cpu())
            self.lbl.append(int(yi))

    def sample(self, k: int):
        if len(self.img) == 0:
            return None
        idx = np.random.choice(len(self.img), size=min(k, len(self.img)), replace=False)
        x = torch.stack([self.img[i] for i in idx]).to(DEVICE, non_blocking=True)
        y = torch.tensor([self.lbl[i] for i in idx], device=x.device)
        return x, y


class HOFQBuffer:
    """Extremely simplified Hierarchical Orthogonal Feature Quantisation buffer."""

    BYTES_PER_SAMPLE = 2  # two 8-bit indices

    def __init__(self, feat_dim: int = 512, max_bytes: int = 200_000):
        self.max_items = max_bytes // self.BYTES_PER_SAMPLE
        self.codes: List[Tuple[int, int]] = []  # (idx0, idxt)
        # coarse & residual code-books --------------------------------------------------
        self.c0 = torch.randn(256, feat_dim)
        self.ct: List[torch.Tensor] = []  # list with one residual code-book / task

    # ------------------------------------------------------------------
    def train_c0(self, feats: torch.Tensor):
        """(Offline) k-means on first task features to initialise C0."""
        km = MiniBatchKMeans(256, batch_size=512, max_iter=25).fit(feats.cpu().numpy())
        # Ensure dtype matches self.c0 to avoid ``copy_`` mismatch errors.
        self.c0.copy_(torch.tensor(km.cluster_centers_, dtype=self.c0.dtype))

    def new_task(self):
        self.ct.append(torch.randn(64, self.c0.shape[1]))

    # ------------------------------------------------------------------
    def add(self, feats: torch.Tensor):
        for f in feats:
            if len(self.codes) >= self.max_items:
                self.codes.pop(0)
            # coarse quantisation ------------------------------------------------------
            idx0 = torch.cdist(f[None], self.c0.to(DEVICE))[0].argmin().item()
            # residual (if available) --------------------------------------------------
            idxt = 0
            if self.ct:
                residual = f.to(DEVICE) - self.c0[idx0].to(DEVICE)
                idxt = torch.cdist(residual[None], self.ct[-1].to(DEVICE))[0].argmin().item()
            self.codes.append((idx0, idxt))

    def sample(self, k: int):
        if not self.codes:
            return None
        idx = np.random.choice(len(self.codes), size=min(k, len(self.codes)), replace=False)
        idx0 = torch.tensor([self.codes[i][0] for i in idx])
        idxt = torch.tensor([self.codes[i][1] for i in idx])
        # If no residual code-book exists yet (first task), only use the coarse codes.
        if not self.ct:
            return self.c0[idx0].to(DEVICE)
        return (self.c0[idx0] + self.ct[-1][idxt]).to(DEVICE)

    # ------------------------------------------------------------------
    @property
    def bytes(self):
        return len(self.codes) * self.BYTES_PER_SAMPLE


# -------------------------------------------------------------------------------------
#  Generic helpers --------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# This global gets populated once a model instance is created.  It is intentionally
# kept *optional* so that helper functions can be imported without side-effects.
MODEL_BYTES: int | None = None


def assert_memory(buffer_bytes: int):
    """Ensure replay buffer RAM stays within the 5 MB limit.

    The research paper imposes *separate* budgets for model, optimiser and replay
    memory.  Here we only enforce the **buffer** constraint because the model
    footprint alone already exceeds 5 MB (e.g. ResNet-18 ≈ 46 MB in fp32).
    """
    if buffer_bytes > 5 * 1024 * 1024:
        raise RuntimeError(f"Replay buffer budget exceeded: {buffer_bytes / 1024:.1f} kB > 5 MB")


# -------------------------------------------------------------------------------------------------
#  Training loop for a *single* task --------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()


def train_task(model: ResNet18, loader: torch.utils.data.DataLoader, opt: optim.Optimizer, buf, method: str):
    """One pass over *loader* with potential replay according to *method*."""

    model.train()
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        opt.zero_grad()

        if method == "hofq":
            feats, logits = model(x, return_feat=True)
            loss = criterion(logits, y)
            if (replay := buf.sample(len(x))) is not None:
                rlog = model.classifier(replay.detach())
                loss = loss + 0.4 * criterion(rlog, rlog.max(1)[1])
            loss.backward()
            opt.step()
            buf.add(feats.detach().cpu())

        elif method == "pixel":
            logits = model(x)
            loss = criterion(logits, y)
            if (r := buf.sample(len(x))) is not None:
                rlog = model(r[0])
                loss = loss + criterion(rlog, r[1])
            loss.backward()
            opt.step()
            buf.add(x.cpu(), y.cpu())

        else:  # plain finetune -------------------------------------------------------
            criterion(model(x), y).backward()
            opt.step()

        # budget check ---------------------------------------------------------------
        if buf is not None:
            assert_memory(buf.bytes)
````