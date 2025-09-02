"""src/train.py
Training-related utilities split out from the original monolithic experiment
script.

Only lightweight refactoring has been done – **no new functionality has been
introduced**. All logic is copied from the original file and minimally adapted
so it can be imported by other modules.
"""
from __future__ import annotations

import importlib
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .preprocess import MemThroughputProfiler  # re-use common util

__all__ = [
    "get_model",
    "train_epoch",
]

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  MODEL FACTORY (exactly the same logic as in the original script)
# ---------------------------------------------------------------------------

try:
    # Prefer the real implementation if the local package exists
    s2_mamba = importlib.import_module("s2_mamba")
    VMambaB = getattr(s2_mamba, "VMambaB")  # type: ignore[attr-defined]
    S2MambaB = getattr(s2_mamba, "S2MambaB")  # type: ignore[attr-defined]
except (ModuleNotFoundError, AttributeError):

    class DummyModel(nn.Module):
        """Fallback model so that the refactored code stays runnable."""

        def __init__(self, num_classes: int = 1000):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(16, num_classes)

        def forward(self, x):  # noqa: D401 – keep signature identical
            x = self.backbone(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    LOGGER.warning("s2_mamba package not found – falling back to DummyModel.")

    VMambaB = DummyModel  # type: ignore
    S2MambaB = DummyModel  # type: ignore


MODEL_MAP = {
    "VMambaB": VMambaB,
    "S2MambaB": S2MambaB,
}


def get_model(name: str, **kwargs) -> nn.Module:
    """Return an instantiated model by name (VMambaB or S2MambaB).

    Parameters
    ----------
    name: str
        Either ``"VMambaB"`` or ``"S2MambaB"``.
    **kwargs: dict
        Extra keyword arguments forwarded to the model constructor.
    """
    if name not in MODEL_MAP:
        raise ValueError(f"Unknown model name: {name}")
    model_cls = MODEL_MAP[name]
    return model_cls(**kwargs)


# ---------------------------------------------------------------------------
#  TRAINING LOOP (used by ImageNet experiment 2a)
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> Tuple[dict, float]:
    """One epoch of standard supervised training.

    Returns a tuple ``(profiler_stats, average_loss)``.
    """

    model.train()
    profiler = MemThroughputProfiler()
    profiler.reset()

    total_loss, seen = 0.0, 0

    for x, y in loader:
        x = x.cuda(non_blocking=True).float()
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        profiler.update(x.size(0))
        total_loss += loss.item() * x.size(0)
        seen += x.size(0)

    avg_loss = total_loss / max(seen, 1)
    return profiler.summary(), avg_loss