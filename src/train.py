"""src/train.py
Training-related modules: model definitions, replay buffer, and the per-task
training loop.
"""
from __future__ import annotations

import math
import random
import time
import os  # Added: required for os.cpu_count()
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Third-party
import geotorch  # Stiefel/orthogonal constraints

# -----------------------------------------------------------------------------
# Device helper – defined once and re-used everywhere
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Backbone – ResNet-18 with Stiefel-regularised last block
# -----------------------------------------------------------------------------
class ResNet18_Stiefel(nn.Module):
    """ResNet-18 whose last convolution block is constrained to be orthogonal.
    The final FC layer is removed so that the network outputs a 512-D feature
    vector that downstream heads can freely use.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # ``weights`` is the recommended API from torchvision>=0.13
        self.resnet = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # Keep feature dimension, replace classifier with identity
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        # Orthogonalise the convolution weights of the last residual block
        geotorch.orthogonal(self.resnet.layer4[-1].conv2, "weight")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.resnet(x)


# -----------------------------------------------------------------------------
# Elastic Feature-Sketch Buffer components
# -----------------------------------------------------------------------------
class OnlineVQEncoder(nn.Module):
    """Light-weight vector-quantiser that operates directly in feature space."""

    def __init__(self, in_dim: int, code_len: int = 8, k: int = 256) -> None:
        super().__init__()
        self.code_len = code_len
        self.k = k
        self.embed = nn.Embedding(k, in_dim)
        self.proj_down = nn.Linear(in_dim, in_dim // 2)
        self.proj_up = nn.Linear(in_dim // 2, in_dim)
        nn.init.uniform_(self.embed.weight, -1, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        # z : [B, D]
        flat = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.weight.T
            + self.embed.weight.pow(2).sum(1)
        )
        return dist.argmin(-1)  # [B]

    # ------------------------------------------------------------------
    def decode(self, codes: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.embed(codes)


class ReservoirItem:
    """Simple struct to hold one buffer element."""

    __slots__ = ("codes", "y", "infl")

    def __init__(self, codes: torch.Tensor, y: int, infl: float) -> None:
        self.codes, self.y, self.infl = codes, y, infl


class EFSBuffer:
    """Gradient-aware reservoir sampling buffer with byte-level budget."""

    def __init__(self, in_dim: int, B_max: int = 1_000_000, code_len: int = 8) -> None:
        self.max_bytes = B_max
        self.code_len = code_len  # bytes per code (each code index → 1 byte)
        self.label_bytes = 2      # uint16 label
        self.bytes_used = 0
        self.items: List[ReservoirItem] = []
        self.encoder = OnlineVQEncoder(in_dim, code_len)

    # ------------------------ helpers ------------------------
    def _sample_cost(self) -> int:
        return self.code_len + self.label_bytes

    # -------------------- public API -------------------------
    def observe(self, feat: torch.Tensor, y: torch.Tensor, infl: torch.Tensor) -> None:
        """Observe *one minibatch* of features and update the reservoir."""
        assert feat.ndim == 2, "Features must be flattened [B, D]"
        codes = self.encoder.encode(feat.cpu())
        for c, yy, inf in zip(codes, y.cpu(), infl.cpu()):
            if self.bytes_used + self._sample_cost() > self.max_bytes:
                # Replace an existing sample if the new one has higher influence
                idx = torch.randint(0, len(self.items), (1,)).item()
                if inf > self.items[idx].infl:
                    self.bytes_used -= self._sample_cost()
                    self.items[idx] = ReservoirItem(c, int(yy), float(inf))
                    self.bytes_used += self._sample_cost()
            else:
                self.items.append(ReservoirItem(c, int(yy), float(inf)))
                self.bytes_used += self._sample_cost()

    # ------------------------------------------------------------------
    def sample(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self) >= k, "Not enough samples in buffer"
        idx = np.random.choice(len(self.items), size=k, replace=False)
        codes = torch.tensor([self.items[i].codes for i in idx], dtype=torch.long)
        ys = torch.tensor([self.items[i].y for i in idx], dtype=torch.long)
        feats = self.encoder.decode(codes).detach()
        return feats, ys

    # ------------------------------------------------------------------
    def wgf_refine(self, step_size: float = 0.1) -> None:
        """One step of SVGD-style refinement in embedding space."""
        if len(self) == 0:
            return
        codes = torch.stack([it.codes for it in self.items])
        embeds = self.encoder.decode(codes)
        grad = torch.autograd.grad(
            outputs=embeds.norm(2, 1).mean(),
            inputs=self.encoder.embed.weight,
            retain_graph=False,
            create_graph=False,
        )[0]
        with torch.no_grad():
            self.encoder.embed.weight += step_size * grad

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.items)


# -----------------------------------------------------------------------------
# Training loop for a single continual-learning task
# -----------------------------------------------------------------------------

def train_single_task(
    backbone: nn.Module,
    classifier: nn.Module,
    buffer: EFSBuffer,
    train_ds,
    val_ds,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    replay_ratio: float = 0.5,
) -> float:
    """Train one task and return validation accuracy."""

    backbone.to(DEVICE)
    classifier.to(DEVICE)
    backbone.train()
    classifier.train()

    optimiser = torch.optim.SGD(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )

    # Deferred import to avoid circular dependency at module level
    from .evaluate import evaluate  # pylint: disable=import-outside-toplevel

    for ep in range(epochs):
        ep_loss, n_samples = 0.0, 0
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            feat = backbone(x)
            out = classifier(feat)
            loss_main = F.cross_entropy(out, y)

            # Influence score for reservoir replacement (norm of logits)
            infl = out.detach().norm(dim=1)
            buffer.observe(feat.detach().cpu(), y.cpu(), infl.cpu())

            # -------------------- Replay --------------------
            if len(buffer) >= int(batch_size * replay_ratio):
                feat_rep, y_rep = buffer.sample(int(batch_size * replay_ratio))
                feat_rep = feat_rep.to(DEVICE)
                y_rep = y_rep.to(DEVICE)
                out_rep = classifier(feat_rep)
                loss_rep = F.cross_entropy(out_rep, y_rep)
                loss = loss_main + loss_rep
            else:
                loss = loss_main

            loss.backward()
            optimiser.step()

            ep_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        sched.step()

        # Every 10 epochs, print and refine
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs} | loss {(ep_loss / n_samples):.4f}")
            buffer.wgf_refine()

    # ------------------------------------------------------------------
    acc = evaluate(backbone, classifier, val_ds)
    return acc
