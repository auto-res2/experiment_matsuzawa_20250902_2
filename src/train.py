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
    def __init__(self, n_cls: int = 100):
        super().__init__()
        from torchvision import models  # local import avoids heavy dependency if unused
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, n_cls)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
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
        self.c0.copy_(torch.tensor(km.cluster_centers_))

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
        return (self.c0[idx0] + self.ct[-1][idxt]).to(DEVICE)

    # ------------------------------------------------------------------
    @property
    def bytes(self):
        return len(self.codes) * self.BYTES_PER_SAMPLE


# -------------------------------------------------------------------------------------
#  Generic helpers --------------------------------------------------------------------
# -------------------------------------------------------------------------------------
MODEL_BYTES: int | None = None  # gets populated once a model is instantiated


def assert_memory(buffer_bytes: int):
    """Ensure total memory < 5 MB (model + optimiser + buffer)."""
    if MODEL_BYTES is None:
        return
    opt_bytes = MODEL_BYTES * 2  # SGD fp32 states m & v
    total = buffer_bytes + MODEL_BYTES + opt_bytes
    assert (
        total <= 5 * 1024 * 1024
    ), f"RAM budget {total / 1024:.1f} kB > 5 MB!"


# -------------------------------------------------------------------------------------------------
#  Training loop for a *single* task --------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()


def train_task(model: ResNet18, loader: torch.utils.data.DataLoader, opt: optim.Optimizer, buf, method: str):
    """One pass over *loader* with potential replay according to *method*."""
    model.train()
    torch.autograd.set_detect_anomaly(True)
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
        assert_memory(buf.bytes if buf else 0)
