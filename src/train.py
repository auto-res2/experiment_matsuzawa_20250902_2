"""src/train.py
Training-related networks, buffers and the per–task training loop.
All heavy lifting that touches the optimiser or back-prop lives here so
other modules can stay light-weight.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Any, Dict
import random, math
import os  # NEW: needed for environment variable patch

# -----------------------------------------------------------------------------
#  Determinism patch for CUDA ≥10.2
# -----------------------------------------------------------------------------
# PyTorch requires a CUBLAS workspace configuration variable to be preset **at
# process start-up** when deterministic algorithms are requested and the code
# path hits GEMM (i.e. almost every deep-net).  Setting the variable here –
# before any CUDA context is initialised – avoids the runtime error:
#   "Deterministic behavior was enabled … but this operation is not deterministic …"
# The chosen value (4096:8) follows the official PyTorch recommendations.
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    # The value must be either ":4096:8" or ":16:8" – the former gives a
    # larger workspace and is marginally faster.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0):
    """Make results exactly reproducible (when determinism is feasible)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # The deterministic flag may reduce performance but prevents subtle races.
    torch.use_deterministic_algorithms(True)

# -----------------------------------------------------------------------------
#  Networks
# -----------------------------------------------------------------------------

class ResNet18Feat(nn.Module):
    """ResNet-18 returning penultimate 512-d features *and* logits.

    The classifier head is kept outside the torchvision backbone to make
    feature access trivial.
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from torchvision import models  # imported lazily to keep import surface minimal
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor, *, return_feat: bool = False):
        feat = self.backbone(x)
        logits = self.classifier(feat.detach() if not self.training else feat)
        return (feat, logits) if return_feat else logits

# -----------------------------------------------------------------------------
#  Replay Buffers
# -----------------------------------------------------------------------------

def _move(t: torch.Tensor, device: torch.device):
    return t.to(device, non_blocking=device.type == "cuda")

class ERBuffer:
    """Standard exemplar replay that stores *raw* image tensors."""
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []

    # ------------------------------------------------------------------
    def add(self, x: torch.Tensor, y: torch.Tensor):
        """Insert a batch into the ring-buffer (FIFO once full)."""
        for img, lbl in zip(x, y):
            if len(self.images) >= self.capacity:
                self.images.pop(0); self.labels.pop(0)
            self.images.append(img.cpu())
            self.labels.append(int(lbl))

    def sample(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.images) == 0:
            raise ValueError("Attempted to sample from an empty ERBuffer.")
        idx = np.random.choice(len(self.images), size=min(k, len(self.images)), replace=False)
        imgs = torch.stack([self.images[i] for i in idx])
        lbls = torch.tensor([self.labels[i] for i in idx])
        return _move(imgs, self.device), _move(lbls, self.device)

# -----------------------------------------------------------------------------
#  Hierarchical Orthogonal Feature Quantisation (HOFQ) replay
# -----------------------------------------------------------------------------

class HOFQBuffer:
    """Feature-space replay with 2-byte codes per sample.

    NOTE: Only the minimal subset needed by Experiment-1 is implemented here.
    """
    def __init__(self, *, feat_dim: int = 512, bytes_per_sample: int = 2,
                 capacity_kb: int = 200, device: torch.device):
        from sklearn.cluster import MiniBatchKMeans  # noqa: slow but only used once
        self.MiniBatchKMeans = MiniBatchKMeans  # keep reference to avoid re-importing
        self.device = device
        self.feat_dim = feat_dim
        self.max_items = (capacity_kb * 1024) // bytes_per_sample
        # Coarse code-book (fixed)
        self.c0 = torch.empty(256, feat_dim, dtype=torch.float32, device=device)
        self.c0_frozen = False
        self.codes: List[Tuple[int, int]] = []  # tuples of (idx0, idx_task)
        self.residual_books: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def _train_coarse_codebook(self, feats: torch.Tensor):
        """One-off k-means initialisation on first 1000 stream samples."""
        mbk = self.MiniBatchKMeans(n_clusters=256, batch_size=512, max_iter=50)
        mbk.fit(feats.cpu().numpy())
        self.c0.copy_(torch.tensor(mbk.cluster_centers_, device=self.device))
        self.c0_frozen = True

    # ------------------------------------------------------------------
    def new_task(self):
        ct = torch.zeros(64, self.feat_dim, device=self.device)
        self.residual_books.append(ct)

    # ------------------------------------------------------------------
    def _encode(self, feat: torch.Tensor) -> Tuple[int, int]:
        # Stage-1: coarse index
        dist0 = torch.cdist(feat.unsqueeze(0), self.c0)[0]
        idx0 = int(dist0.argmin())
        residual = feat - self.c0[idx0]
        # Stage-2: task-specific residual code-book
        ct = self.residual_books[-1]
        if ct.abs().sum() == 0:  # lazy random warm-start
            rand_idx = np.random.choice(self.c0.shape[0], size=64, replace=False)
            ct.copy_(self.c0[rand_idx])
        distt = torch.cdist(residual.unsqueeze(0), ct)[0]
        idxt = int(distt.argmin())
        return idx0, idxt

    # ------------------------------------------------------------------
    def add(self, feats: torch.Tensor):
        for f in feats:
            if len(self.codes) >= self.max_items:
                self.codes.pop(0)
            self.codes.append(self._encode(f.detach()))

    # ------------------------------------------------------------------
    def sample(self, k: int) -> torch.Tensor:
        if len(self.codes) == 0:
            raise ValueError("Attempted to sample from an empty HOFQBuffer.")
        idx = np.random.choice(len(self.codes), size=min(k, len(self.codes)), replace=False)
        idx0 = torch.tensor([self.codes[i][0] for i in idx], device=self.device)
        idxt = torch.tensor([self.codes[i][1] for i in idx], device=self.device)
        base = self.c0[idx0]
        ct = self.residual_books[-1][idxt]
        return base + ct

# -----------------------------------------------------------------------------
#  Per-task training loop
# -----------------------------------------------------------------------------

def train_single_task(model: nn.Module, loader: torch.utils.data.DataLoader,
                      optimizer: optim.Optimizer, criterion: nn.Module,
                      method: str, buffer: Any | None = None, task_id: int = 0,
                      *, device: torch.device):
    """One pass over *one* task dataset with optional replay."""

    model.train()
    pbar = loader
    for x, y in pbar:
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad()

        if method == "hofq":
            feats, logits = model(x, return_feat=True)
            loss_cls = criterion(logits, y)
            replay_loss = torch.tensor(0.0, device=device)
            if buffer and len(buffer.codes):
                replay_feats = buffer.sample(k=len(x))
                replay_logits = model.classifier(replay_feats.detach())
                pseudo_labels = replay_logits.detach().max(1)[1]
                replay_loss = criterion(replay_logits, pseudo_labels)
            loss = loss_cls + 0.4 * replay_loss
            loss.backward()
            optimizer.step()
            # sparsify gradients in lower backbone layers
            for n, p in model.backbone.named_parameters():
                if ("layer1" in n or "layer2" in n) and p.grad is not None:
                    p.grad.zero_()
            buffer.add(feats.detach().cpu())

        elif method.startswith("er"):
            logits = model(x)
            loss = criterion(logits, y)
            if buffer and len(buffer.images):
                bx, by = buffer.sample(k=len(x))
                blogits = model(bx)
                loss += criterion(blogits, by)
            loss.backward(); optimizer.step();
            buffer.add(x.cpu(), y.cpu())

        elif method == "finetune":
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
        else:
            raise ValueError(f"Unsupported method: {method}")