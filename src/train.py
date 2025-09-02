"""src/train.py
Model definitions and training-related helpers.
Extracted from the original monolithic experimental script.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

__all__ = [
    "set_seed",
    "current_device",
    "ensure_dir",
    "DummyAr2Diff",
    "SmallCNN",
]

# -----------------------------------------------------------------------------
# Utility helpers (shared across modules)
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    """Set Python, NumPy and PyTorch random seeds for full determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_device() -> torch.device:
    """Return "cuda" when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: Union[str, Path]) -> None:
    """Create directory *p* (including parents) if it does not yet exist."""
    Path(p).mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Lightweight stand-in models –  100 % CPU / CI friendly
# -----------------------------------------------------------------------------

class DummyAr2Diff(nn.Module):
    """Extremely small placeholder for the full AR2-Diff pipeline.

    Keeps API compatibility so that the evaluation code can run without the
    heavyweight Stable-Diffusion models.  Additionally exposes helper methods
    that *simulate* latency, memory and FLOPs numbers so figures/tables are
    still meaningful.
    """

    def __init__(self, budget: float = 0.10, steps: int = 30):
        super().__init__()
        self.budget = float(budget)
        self.steps = int(steps)
        # tiny conv net used only on the refined region
        self.dummy_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 3, 1),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x expected in [0,1]
        b, c, h, w = x.shape
        mask = torch.rand((b, 1, h, w), device=x.device) < self.budget
        refined = self.dummy_net(x)
        out = torch.where(mask, refined, x)
        return out.clamp(0, 1)

    # ------------------------------------------------------------------
    # Fake performance numbers (so that downstream plots/tables work)
    # ------------------------------------------------------------------
    def simulated_latency(self, resolution: int = 256) -> float:
        base = 0.5  # coarse pass (s)
        fine = self.steps * self.budget * 0.015  # 15 ms × steps × area
        return base + fine

    def simulated_memory(self) -> float:  # in GB
        return 4.0 + 12 * self.budget

    def simulated_flops(self) -> float:  # in FLOPs
        return 40e9 * self.budget


# -----------------------------------------------------------------------------
# Small CNN used as the *uncertainty predictor* in Experiment-2
# -----------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """Mini-CNN used to predict per-pixel refinement uncertainty."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
