"""src/train.py
All routines that deal with model construction and training.
"""
from __future__ import annotations
import random, time
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (kept for potential future use)
import timm

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS -------------------------------------------------------------
# These are imported by other modules, therefore keep them here to avoid a
# circular-import maze.
# -----------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3]
BF16_ENABLED = torch.cuda.is_available()
BASELINE_NAME = "vmamba_tiny_patch16_224"   # timm id

# -----------------------------------------------------------------------------
# 0.  REPRODUCIBILITY HELPERS --------------------------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set RNG seeds for python, numpy and torch (CPU / CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 95 % confidence interval helper (imported by evaluate)
ci95 = lambda x: (np.mean(x), 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (x[0], 0.0)

# -----------------------------------------------------------------------------
# 1.  MEMORY-UTILITY -----------------------------------------------------------
# -----------------------------------------------------------------------------

def peak_ram_gb() -> float:
    """Return *peak* GPU memory (in GB) measured so far on the active CUDA device."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 3

# -----------------------------------------------------------------------------
# 2.  MODEL BUILDERS -----------------------------------------------------------
# -----------------------------------------------------------------------------
class FlashWrapper(nn.Module):
    """A very small wrapper that mimics the proposed Flash-SSM chunking strategy.
    It uses `torch.utils.checkpoint` to drop activations outside a window of
    size `self.chunk`.
    """

    def __init__(self, blk: nn.Module, chunk: int):
        super().__init__()
        self.blk = blk
        self.chunk = chunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if x.shape[1] <= self.chunk:
            return torch.utils.checkpoint.checkpoint(self.blk, x)
        pieces = []
        for s in range(0, x.shape[1], self.chunk):
            pieces.append(torch.utils.checkpoint.checkpoint(self.blk, x[:, s : s + self.chunk]))
        return torch.cat(pieces, 1)


def _convert_flash(model: nn.Module, chunk: int = 256) -> nn.Module:
    """Recursively replace the *mixer* inside each VMamba block with a memory
    efficient `FlashWrapper`.
    """

    for _name, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for i, blk in enumerate(module):
                if hasattr(blk, "mixer"):
                    blk.mixer = FlashWrapper(blk.mixer, chunk)
        _convert_flash(module, chunk)
    return model


def build_model(impl: str = "baseline", chunk: int = 256) -> nn.Module:
    """Factory that returns either the *baseline* VMamba model or the *flash*
    variant with chunked scan.
    """

    model = timm.create_model(BASELINE_NAME, pretrained=False)
    if impl == "flash":
        model = _convert_flash(model, chunk)
    return model

# -----------------------------------------------------------------------------
# 3.  TRAINING LOOP ------------------------------------------------------------
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,  # type: ignore
    optimiser: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    total_epochs: int,
    log_mid: int = 150,
) -> float:
    """Perform one training epoch and return *mid-epoch* peak RAM (GB)."""

    criterion = nn.CrossEntropyLoss()
    model.train()
    batches = len(loader)
    torch.cuda.reset_peak_memory_stats()

    mid_peak = 0.0
    for i, (img, tgt) in enumerate(loader):
        img = img.cuda(non_blocking=True)
        tgt = tgt.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16_ENABLED):
            out = model(img)
            loss = criterion(out, tgt)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        optimiser.zero_grad(set_to_none=True)

        if i == log_mid:
            mid_peak = peak_ram_gb()
    return mid_peak
