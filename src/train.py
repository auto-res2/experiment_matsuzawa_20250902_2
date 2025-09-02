"""src/train.py
Training utilities and model definitions extracted from the original monolithic
script.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 – might be useful for extensions
from torch.utils.data import DataLoader

__all__ = [
    "DummyVMambaTiny",
    "DummyFlashSSMTiny",
    "train_one_epoch",
]


# -----------------------------------------------------------------------------
# 1.  Simplified placeholder models
# -----------------------------------------------------------------------------


class DummyVMambaTiny(nn.Module):
    """A tiny CNN pretending to be the baseline VMamba-Tiny.

    We keep large feature maps so that the memory-benchmark logic in the
    evaluation module still produces non-trivial numbers.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(4):
            blocks += [
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
                nn.ReLU(inplace=True),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avg(x).flatten(1)
        return self.fc(x)


class DummyFlashSSMTiny(nn.Module):
    """A memory-efficient counterpart that uses fewer activations by design."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, 2, 1),  # stride 2 – halves H and W
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        blocks = []
        for _ in range(4):
            blocks += [
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1),
                nn.GELU(),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avg(x).flatten(1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# 2.  Single-epoch training routine
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:
    """Very short training loop (single epoch)."""

    device = next(model.parameters()).device
    model.train()
    criterion = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(imgs)
            loss = criterion(out, labels)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
