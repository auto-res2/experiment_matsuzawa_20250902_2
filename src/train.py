"""src/train.py
Training-related modules: model definitions, loss functions, training loop, and accuracy helper.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torchvision.models as tv_models
import timm
from tqdm import tqdm

__all__ = [
    "FeatureBackbone",
    "SCaRINet",
    "mse_invariance",
    "instance_dro_weight",
    "accuracy",
    "train_epoch",
]


# -----------------------------------------------------------------------------
#  Model factory
# -----------------------------------------------------------------------------
class FeatureBackbone(nn.Module):
    """Wrapper that exposes a feature extractor with a known output dimension."""

    def __init__(self, arch: str, pretrained: bool = True):
        super().__init__()
        self.arch = arch
        if arch == "resnet50":
            net = tv_models.resnet50(pretrained=pretrained)
            self.feature_dim = net.fc.in_features  # type: ignore[attr-defined]
            net.fc = nn.Identity()
            self.backbone = net
        elif arch == "vit_base_patch16_224":
            net = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
            self.feature_dim = net.head.in_features  # type: ignore[attr-defined]
            net.reset_classifier(0)
            self.backbone = net
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return feature representation of the input batch."""
        return self.backbone(x)


class SCaRINet(nn.Module):
    """Backbone + projection head + classifier used in SCaRI."""

    def __init__(self, arch: str, num_classes: int):
        super().__init__()
        self.encoder = FeatureBackbone(arch)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )
        self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor):  # noqa: D401
        """Return (feature, projection, logits)."""
        feat = self.encoder(x)
        proj = self.projector(feat)
        logits = self.classifier(feat)
        return feat, proj, logits


# -----------------------------------------------------------------------------
#  Losses & metrics
# -----------------------------------------------------------------------------

def mse_invariance(f_anchor: torch.Tensor, f_cf: torch.Tensor) -> torch.Tensor:
    """Representation-level invariance loss (MSE)."""
    return F.mse_loss(f_anchor.unsqueeze(1), f_cf)


def instance_dro_weight(losses: torch.Tensor, eta: float = 0.2) -> torch.Tensor:
    """Adaptive weighting similar to Group-DRO (soft-max over cf losses)."""
    weights = torch.softmax(losses / eta, dim=1)
    return (weights * losses).sum()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    _, pred = torch.max(logits, 1)
    return (pred == y).float().mean().item() * 100.0


# -----------------------------------------------------------------------------
#  Training loop
# -----------------------------------------------------------------------------

def train_epoch(
    model: SCaRINet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    cit_tau: float,
    eta: float,
) -> Tuple[float, float]:
    """Run one training epoch and return (avg_loss, avg_accuracy)."""

    model.train()
    ce_meter, acc_meter = [], []

    for batch in tqdm(loader, desc="train", leave=False):
        # Unpack batch ---------------------------------------------------------
        x, y, _, cf_imgs = batch  # metadata not used
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        cf_imgs = cf_imgs.to(device, non_blocking=True) if cf_imgs is not None else None

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            feat, proj, logits = model(x)
            ce_loss = F.cross_entropy(logits, y)
            total_loss = ce_loss

            # Counterfactual branch -------------------------------------------
            if cf_imgs is not None:
                B, K = cf_imgs.shape[0], cf_imgs.shape[1]
                cf_imgs_flat = cf_imgs.view(B * K, *cf_imgs.shape[2:])
                cf_feat, cf_proj, cf_logits = model(cf_imgs_flat)
                cf_feat = cf_feat.view(B, K, -1)
                cf_logits = cf_logits.view(B, K, -1)

                # Representation invariance
                inv_loss = mse_invariance(feat, cf_feat)

                # Prediction consistency
                y_repeat = y.unsqueeze(1).repeat(1, K).view(-1)
                pred_consistency = F.cross_entropy(
                    cf_logits.view(-1, cf_logits.shape[-1]), y_repeat
                )

                # Instance-level DRO weighting
                with torch.no_grad():
                    per_cf_loss = F.cross_entropy(
                        cf_logits.view(-1, cf_logits.shape[-1]),
                        y_repeat,
                        reduction="none",
                    ).view(B, K)
                dro_loss = instance_dro_weight(per_cf_loss, eta)

                total_loss = ce_loss + inv_loss * cit_tau + pred_consistency + dro_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ce_meter.append(total_loss.item())
        acc_meter.append(accuracy(logits.detach(), y))

    return float(np.mean(ce_meter)), float(np.mean(acc_meter))
