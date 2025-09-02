"""src/train.py
Model definitions and training utilities extracted from the original monolithic
script.
"""
from __future__ import annotations

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 – may be useful for user extension

# ---------------------------------------------------------------------------- #
# Reproducibility helpers
# ---------------------------------------------------------------------------- #

SEEDS: List[int] = [0, 1, 2, 3]


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch (CPU & CUDA) for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------- #
# 1. Mamba layers & blocks (baseline + Flash-SSM style)
# ---------------------------------------------------------------------------- #


class SimpleMambaLayer(nn.Module):
    """Minimal Mamba layer (full-sequence scan).  Works on B×L×C tensors."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.A = nn.Parameter(torch.randn(dim))
        self.B = nn.Parameter(torch.randn(dim))
        self.C = nn.Parameter(torch.randn(dim))
        self.D = nn.Parameter(torch.randn(dim))
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B L C
        B, L, C = x.shape
        h = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = self.A * h + self.B * x[:, t, :]
            y = self.C * h + self.D * x[:, t, :]
            outs.append(y)
        y = torch.stack(outs, dim=1)
        return self.activation(y)


class ChunkedMambaLayer(nn.Module):
    """Chunked prefix scan (Flash-SSM idea) – discards inner activations."""

    def __init__(self, dim: int, chunk: int = 256):
        super().__init__()
        self.dim = dim
        self.chunk = chunk
        self.A = nn.Parameter(torch.randn(dim))
        self.B = nn.Parameter(torch.randn(dim))
        self.C = nn.Parameter(torch.randn(dim))
        self.D = nn.Parameter(torch.randn(dim))
        self.activation = nn.GELU()

    def _scan_chunk(self, x: torch.Tensor, h0: torch.Tensor):
        B, Lc, C = x.shape
        h = h0
        outs = []
        for t in range(Lc):
            h = self.A * h + self.B * x[:, t, :]
            y = self.C * h + self.D * x[:, t, :]
            outs.append(y)
        y = torch.stack(outs, dim=1)
        # Detach to avoid storing full history and save memory
        return y, h.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B L C
        B, L, C = x.shape
        h = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outputs = []
        for start in range(0, L, self.chunk):
            y, h = self._scan_chunk(x[:, start : start + self.chunk, :], h)
            outputs.append(y)
        y = torch.cat(outputs, dim=1)
        return self.activation(y)


class MambaBlock(nn.Module):
    """Baseline residual block with full-sequence Mamba layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mamba = SimpleMambaLayer(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B L C
        x = x + self.mamba(self.ln(x))
        x = x + self.ffn(x)
        return x


class FlashMambaBlock(nn.Module):
    """Flash-SSM block – chunked layer + reversible residual coupling."""

    def __init__(self, dim: int, chunk_len: int = 256):
        super().__init__()
        self.chunk_len = chunk_len
        self.mamba = ChunkedMambaLayer(dim, chunk=chunk_len)
        self.ln = nn.LayerNorm(dim)
        # Lightweight mixing layer; avoids degeneration when using a reversible pattern
        self.mix = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B L C
        # Split channels into two halves (simplified reversible coupling)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1 + self.mamba(self.ln(x2))
        y2 = x2  # identity – can be reconstructed in backward if needed
        y = torch.cat([y1, y2], dim=-1)
        return y + self.mix(y)


# ---------------------------------------------------------------------------- #
# 2. Tiny Vision backbone built from the blocks above
# ---------------------------------------------------------------------------- #


class TinyVisionMamba(nn.Module):
    """Patch-embedding → N blocks → CLS-head. Block class chooses baseline/flash."""

    def __init__(
        self,
        block_cls,
        depth: int = 4,
        dim: int = 128,
        num_classes: int = 1000,
        chunk_len: int | None = 256,
    ):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=8, stride=8)  # 224→28×28=784 tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 784, dim))
        self.blocks = nn.ModuleList(
            [
                block_cls(dim, chunk_len) if block_cls is FlashMambaBlock else block_cls(dim)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B 3 H W
        B = x.size(0)
        x = self.patch(x)  # B  C H/8 W/8
        x = x.flatten(2).transpose(1, 2)  # B  L  C  where L=784
        cls = self.cls_token.expand(B, -1, -1)  # B 1 C
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ---------------------------------------------------------------------------- #
# 3. One-epoch training helper
# ---------------------------------------------------------------------------- #

def train_one_epoch(model: nn.Module, loader, optim) -> float:
    """Runs a *single* epoch; returns elapsed time in seconds."""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    model.train()

    start_t = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_t = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if start_t is not None:
        start_t.record()
    else:
        import time as _time
        wall_start = _time.perf_counter()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optim.step()

    if end_t is not None:
        end_t.record()
        torch.cuda.synchronize()
        elapsed_ms = start_t.elapsed_time(end_t)
        return elapsed_ms / 1e3  # seconds
    else:
        import time as _time
        return _time.perf_counter() - wall_start
