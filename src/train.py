"""
train.py
Core model architectures and training utilities for the SCaRI experiments.
All training-related code is centralised here so it can be imported from
other modules (e.g. src.main).
"""
from __future__ import annotations

import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as tv_models
import timm
from tqdm import tqdm

__all__ = [
    "seed_everything",
    "SCaRINet",
    "train_one_epoch",
    "accuracy_from_logits",
    "instance_dro_weight",
]

# -----------------------------------------------------------------------------
# Deterministic behaviour helper
# -----------------------------------------------------------------------------
TORCH_DETERMINISTIC = True


def seed_everything(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if TORCH_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
#   Models – ResNet-50 & ViT-B/16 with projection head & classifier
# -----------------------------------------------------------------------------

def _get_backbone(arch: str):
    if arch == "resnet50":
        net = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim
    if arch == "vit_b16":
        net = timm.create_model("vit_base_patch16_224", pretrained=True)
        feat_dim = net.embed_dim  # type: ignore[attr-defined]
        net.reset_classifier(0)
        return net, feat_dim
    raise ValueError(f"Unsupported arch {arch}")


class SCaRINet(nn.Module):
    """Encoder + projector + classifier used in all experiments."""

    def __init__(self, arch: str, n_classes: int):
        super().__init__()
        self.encoder, feat_dim = _get_backbone(arch)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):  # type: ignore[override]
        feat = self.encoder(x)
        proj = self.projector(feat)
        logits = self.classifier(feat)
        return feat, proj, logits


# -----------------------------------------------------------------------------
#   Metrics & training utilities
# -----------------------------------------------------------------------------

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy in percentage."""
    return (logits.argmax(1) == y).float().mean().item() * 100.0


def instance_dro_weight(cf_losses: torch.Tensor, eta: float = 0.2) -> torch.Tensor:
    """Instance-level DRO weighting used in SCaRI training."""
    w = torch.softmax(cf_losses / eta, dim=1)
    return (w * cf_losses).sum()


# -----------------------------------------------------------------------------
#   One-epoch trainer (SCaRI variant)
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    lambda_inv: float = 1.0,
    eta: float = 0.2,
) -> Dict[str, float]:
    """Train a single epoch with SCaRI losses.

    If a batch does not include counterfactual images an assertion will be
    raised – this is by design to prevent silent misuse of the method.
    """

    model.train()
    ce_losses, accs = [], []

    for x, y, _meta, cf_imgs in tqdm(loader, desc="train", leave=False):
        # --------------------------------------------------
        # Mandatory CF branch – abort if missing (policy)
        # --------------------------------------------------
        assert (
            cf_imgs is not None
        ), "CF branch inactive – abort (counterfactuals missing)"
        B, K = cf_imgs.shape[0], cf_imgs.shape[1]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        cf_imgs = cf_imgs.view(B * K, *cf_imgs.shape[2:]).to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast():
            feat, _proj, logits = model(x)
            feat_cf, _proj_cf, logits_cf = model(cf_imgs)
            feat_cf = feat_cf.view(B, K, -1)
            logits_cf = logits_cf.view(B, K, -1)

            ce_orig = F.cross_entropy(logits, y)
            y_rep = y.unsqueeze(1).repeat(1, K).view(-1)
            ce_cf = F.cross_entropy(logits_cf.view(-1, logits_cf.shape[-1]), y_rep)
            inv = F.mse_loss(feat.unsqueeze(1), feat_cf)
            with torch.no_grad():
                per_cf = F.cross_entropy(
                    logits_cf.view(-1, logits_cf.shape[-1]), y_rep, reduction="none"
                ).view(B, K)
            dro = instance_dro_weight(per_cf, eta)
            loss = ce_orig + ce_cf + lambda_inv * inv + dro

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        ce_losses.append(loss.item())
        accs.append(accuracy_from_logits(logits.detach(), y))

    return {
        "train_loss": float(np.mean(ce_losses)),
        "train_acc": float(np.mean(accs)),
    }
