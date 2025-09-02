"""
train.py – Model, training utilities and MUCD core components
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import timm
from scipy.fft import fftshift, ifftshift

# -----------------------------------------------------------------------------
# Device & dtype helpers
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seed on Python / NumPy / PyTorch (both CPU & GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Accuracy helper (shared between train / evaluate)
# -----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item() * 100.0

# -----------------------------------------------------------------------------
# MUCD core – Environment discovery & perturbations
# -----------------------------------------------------------------------------


class EnvCluster:
    """Unsupervised environment discovery via mini-batch spectral clustering."""

    def __init__(self, emb_dim: int = 128, num_clusters: int = 8):
        from sklearn.cluster import SpectralClustering  # local import to avoid heavy cost when unused

        self.emb_dim = emb_dim
        self.num_clusters = num_clusters
        # Avoid downloading weights – use random initialisation (pretrained=False)
        self.encoder = timm.create_model("resnet18", pretrained=False, num_classes=0).to(DEVICE)
        self.encoder.eval()
        # projection head
        self.proj = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, emb_dim)
        ).to(DEVICE)
        for p in self.proj.parameters():
            p.requires_grad = False
        self._clusterer = None  # will be fitted each epoch
        self._SpectralClustering = SpectralClustering  # keep reference

    @torch.no_grad()
    def _embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        z = self.proj(f)
        z = F.normalize(z, dim=1)
        # concatenate log power spectrum
        x_fft = torch.fft.rfftn(x.float(), dim=(-2, -1))
        power = torch.log1p(torch.abs(x_fft))
        power = power.mean(dim=(2, 3))  # avg over H,W & channels
        feat = torch.cat([z, power], dim=1)
        return feat.cpu()

    def fit(self, loader: DataLoader):
        """Fit clustering model using all samples from the provided loader."""
        feats = []
        for xb, _ in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            feats.append(self._embed_batch(xb))
        feats = torch.cat(feats, dim=0).numpy()
        # fit spectral clustering each epoch (fresh model)
        self._clusterer = self._SpectralClustering(
            n_clusters=self.num_clusters,
            affinity="nearest_neighbors",
            n_neighbors=10,
            random_state=0,
        ).fit(feats)

    @torch.no_grad()
    def assign(self, x: torch.Tensor) -> torch.Tensor:
        assert self._clusterer is not None, "Clusterer not fitted – call `.fit()` first."
        feats = self._embed_batch(x)
        labels = self._clusterer.predict(feats.numpy())
        return torch.tensor(labels, device=x.device)


class DualPerturber(nn.Module):
    """Combined semantic and frequency perturbation module used by MUCD."""

    def __init__(self):
        super().__init__()
        import torchvision  # local import to minimise global dependency

        self.erase = torchvision.transforms.RandomErasing(
            p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        import random
        import torch

        # Semantic: random erase (proxy for CLIP+SAM masking)
        x_sem = torch.stack([self.erase(img.clone()) for img in x])
        # Frequency: swap random 4×4 Fourier blocks with another image in batch
        x_freq = self._frequency_swap(x)
        # Mix two views randomly
        mask = torch.rand(len(x), device=x.device) < 0.5
        out = torch.where(mask[:, None, None, None], x_sem, x_freq)
        return out

    @staticmethod
    def _frequency_swap(x: torch.Tensor) -> torch.Tensor:
        import random

        b, c, h, w = x.shape
        Xf = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
        Xf = fftshift(Xf, dim=(-2, -1))
        # choose random 4×4 region per sample
        for i in range(b):
            h0 = random.randint(0, h // 2 - 4)
            w0 = random.randint(0, w // 2 - 4)
            idx = random.randint(0, b - 1)
            block = Xf[idx, :, h0 : h0 + 4, w0 : w0 + 4].clone()
            Xf[i, :, h0 : h0 + 4, w0 : w0 + 4] = block
        Xf = ifftshift(Xf, dim=(-2, -1))
        xf = torch.fft.irfft2(Xf, s=(h, w), dim=(-2, -1), norm="ortho")
        xf = xf.clamp(0, 1).type_as(x)
        return xf

# -----------------------------------------------------------------------------
# Loss function – Causal Effect Regularisation (CER)
# -----------------------------------------------------------------------------

def cer_loss(
    logits_orig: torch.Tensor, logits_pert: torch.Tensor, env_id: torch.Tensor
) -> torch.Tensor:
    """Implementation of CER loss described in the paper."""
    diff = logits_orig.detach() - logits_pert  # stop gradient through perturbation branch
    ce = (diff ** 2).mean(dim=1)
    # normalise within each environment cluster
    unique_env = env_id.unique()
    loss = 0.0
    for e in unique_env:
        mask = env_id == e
        loss += ce[mask].mean()
    return loss / len(unique_env)

# -----------------------------------------------------------------------------
# Experiment runner (training & validation loop)
# -----------------------------------------------------------------------------


class ExperimentRunner:
    """Wrapper that owns model, optimiser, and training / validation loops."""

    def __init__(
        self,
        exp_name: str,
        num_classes: int,
        lr: float = 3e-4,
        epochs: int = 2,  # shortened by default for quick sanity-check
    ):
        self.exp_name = exp_name
        self.epochs = epochs

        # Backbone – ImageNet-pretrained ResNet-50 (set pretrained=False to avoid net access)
        self.model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes).to(
            DEVICE
        )

        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05
        )
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # MUCD specific components
        self.clusterer = EnvCluster(num_clusters=8)
        self.perturber = DualPerturber().to(DEVICE)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, loader_train: DataLoader, loader_val: DataLoader):
        best_acc = 0.0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            self.model.train()

            # 1) environment re-estimation each epoch
            self.clusterer.fit(loader_train)

            # 2) iterate over training mini-batches
            for xb, yb in loader_train:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                env_id = self.clusterer.assign(xb)
                xb_pert = self.perturber(xb)

                with autocast(enabled=torch.cuda.is_available()):
                    logits = self.model(xb)
                    logits_p = self.model(xb_pert)
                    loss_erm = F.cross_entropy(logits, yb)
                    loss_cer = cer_loss(logits, logits_p, env_id)
                    loss = loss_erm + loss_cer

                self.optim.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

            # epoch-level validation
            val_acc = self.evaluate(loader_val)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f"{self.exp_name}_best.pt")
            print(f"Validation Acc: {val_acc:.2f}% – Best: {best_acc:.2f}%")

        # restore best checkpoint for subsequent testing
        ckpt = Path(f"{self.exp_name}_best.pt")
        if ckpt.exists():
            self.model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # ------------------------------------------------------------------
    # Validation / testing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        accs: List[float] = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            logits = self.model(xb)
            accs.append(accuracy(logits, yb))
        return float(np.mean(accs))
