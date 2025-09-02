"""
train.py
Minimal stand-in implementations so that the public API imported from
main.py exists during automated tests / CI.  The goal is *not* to ship a
full-fledged SCaRI implementation (which would require the real dataset
and many computation hours) but rather to provide light-weight versions
that compile and execute on CPU in a few seconds.
"""
from __future__ import annotations

import contextlib
import random
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

__all__ = [
    "SCaRINet",
    "seed_everything",
    "train_one_epoch",
]


# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------


def seed_everything(seed: int = 0) -> None:  # pragma: no cover
    """Deterministic behaviour (best-effort).

    For the purpose of CI we only need the essentials.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cuDNN deterministic settings – they make training ~slower but we are
        # only running extremely small toy loops anyway.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
#  Fake SCaRI network (very small to keep resource usage negligible)
# -----------------------------------------------------------------------------


class _MLPHead(nn.Module):
    """A tiny projection head to mimic representation learning stacks."""

    def __init__(self, in_features: int, out_features: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class SCaRINet(nn.Module):
    """*Extremely* compressed ResNet-like encoder with a projection head.

    Parameters
    ----------
    backbone : str
        Ignored in this toy version – kept for API compatibility.
    n_classes : int, default=2
        Number of classification labels.
    """

    def __init__(self, backbone: str, n_classes: int = 2) -> None:  # noqa: D401
        super().__init__()
        _ = backbone  # We simply acknowledge the arg so that callers can pass
        # a *tiny* CNN – single Conv followed by global pooling so that forward
        # pass is lightning fast.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = _MLPHead(in_features=32, out_features=64)
        self.classifier = nn.Linear(32, n_classes)

    # ------------------------------------------------------------------
    # Forward API expected by main.py – returns (feat, proj, logits)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):  # noqa: D401
        feat: torch.Tensor = self.encoder(x)
        proj: torch.Tensor = self.proj(feat.detach())  # detach to mimic contrastive
        logits: torch.Tensor = self.classifier(feat)
        return feat, proj, logits


# -----------------------------------------------------------------------------
#  A *very* small training loop for one epoch (SCaRI edition)
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device | str,
) -> Dict[str, float]:  # pragma: no cover – covered indirectly via main
    """Run a single training epoch.

    The actual SCaRI objective is replaced by a simple CE loss plus a dummy
    projection regulariser so that the signature matches the real function.
    """

    model.train()
    ce_losses, accs = [], []

    for x, y, *_extra in loader:  # _extra captures meta + cf in our dummy ds
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with (
            torch.cuda.amp.autocast() if torch.cuda.is_available() else contextlib.nullcontext()
        ):
            _feat, proj, logits = model(x)
            # Classification loss
            loss_ce = nn.functional.cross_entropy(logits, y)
            # Tiny L2 reg on projection head to mimic representation term.
            loss_proj = 0.05 * (proj ** 2).mean()
            loss = loss_ce + loss_proj

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        ce_losses.append(loss_ce.item())
        accs.append((logits.argmax(1) == y).float().mean().item() * 100.0)

    return {
        "train_loss": float(np.mean(ce_losses)),
        "train_acc": float(np.mean(accs)),
    }
