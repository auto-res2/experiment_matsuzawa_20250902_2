"""src/train.py
Training-related components: model, losses, accuracy helpers and one-epoch
training loop.
"""
from __future__ import annotations

import random
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast

# Third-party (assume already installed by the entry script)
import timm
from diffusers import AutoencoderKL

# ---------------------------------------------------------------------
# Reproducibility & device
# ---------------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------

def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy in %."""
    return (pred == target).float().mean().item() * 100.0


def worst_group_accuracy(pred: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> float:
    """Worst (minimum) accuracy across groups in %."""
    import pandas as pd  # local import to keep dependency graph simple

    df = pd.DataFrame({"pred": pred.cpu(), "y": target.cpu(), "g": group.cpu()})
    group_accs = df.groupby("g").apply(lambda d: (d.pred == d.y).mean())
    return (group_accs.min() * 100.0).item()

# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """Standard contrastive Info-NCE loss (NT-Xent)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:  # type: ignore
        # q, k: [B, D]  – assume already L2-normalised
        logits = q @ k.T / self.temperature
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

# ---------------------------------------------------------------------
# Main CC-LiDAR model wrapper
# ---------------------------------------------------------------------

class CCLiDAR(nn.Module):
    """Classifier augmented with latent diffusion disentanglement & regularisers."""

    def __init__(self, backbone: nn.Module, cfg: SimpleNamespace):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        # Frozen Stable-Diffusion VAE encoder/decoder
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
        )
        self.vae.requires_grad_(False)

        latent_dim = (
            self.vae.config.latent_channels * (cfg.latent_h // 8) * (cfg.latent_w // 8)
        )
        proj_dim = 512
        self.content_head = nn.Linear(latent_dim, proj_dim, bias=False)
        self.context_head = nn.Linear(latent_dim, proj_dim, bias=False)
        self.info_nce = InfoNCELoss(cfg.tau)

    # ------------------   Latent helpers   ------------------
    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to SD latent and flatten."""
        with autocast(enabled=self.cfg.fp16):
            posterior = self.vae.encode(x).latent_dist  # N(μ, σ)
            z = posterior.mean  # [B, 4, H/8, W/8]
        return z.flatten(start_dim=1)  # [B, C*H*W]

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.backbone(x)

    # ------------------   Forward pass   ------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # type: ignore
        """Forward pass returning the composite loss and logging info."""
        x: torch.Tensor = batch["x"].to(device)
        y: torch.Tensor = batch["y"].to(device)

        # -- Encode latents (no grad)
        with torch.no_grad():
            z = self.encode_latent(x)
        z_c = self.content_head(z)
        z_k = self.context_head(z)

        # -- Main classifier logits & CE
        logits = self.forward_backbone(x)
        loss_cls = F.cross_entropy(logits, y)

        # ------------- InfoNCE (view-invariance) -------------
        x2 = batch["x2"].to(device)  # supplied by training loop
        with torch.no_grad():
            z2 = self.encode_latent(x2)
        z2_c = self.content_head(z2)
        z2_k = self.context_head(z2)
        nce_c = self.info_nce(F.normalize(z_c, dim=-1), F.normalize(z2_c, dim=-1))
        nce_k = -self.info_nce(F.normalize(z_k, dim=-1), F.normalize(z2_k, dim=-1))
        loss_nce = nce_c + nce_k

        # ------------- Counterfactual ACE regulariser -------------
        perm = torch.randperm(z_k.size(0))
        z_k_perm = z_k[perm]
        with torch.no_grad():
            # crude "concatenate & reshape" back to latent grid
            full_latent = torch.cat([z_c, z_k_perm], dim=1)
            full_latent = full_latent.view(
                x.size(0), 4, self.cfg.latent_h // 8, self.cfg.latent_w // 8
            )
            x_cf = self.vae.decode(full_latent).sample  # type: ignore
        with autocast(enabled=self.cfg.fp16):
            logits_cf = self.forward_backbone(x_cf)
        ace = (logits - logits_cf).pow(2).mean(dim=-1)  # per-sample squared diff
        loss_ace_reg = self.cfg.lambda_ce * ace.mean()

        # ---------------- Fourier consistency (HFF) ----------------
        if self.cfg.lambda_hff > 0:
            x_fft = torch.fft.rfft2(x, norm="ortho")
            amp, phase = x_fft.abs(), x_fft.angle()
            phase_perm = phase[perm]
            x_ifft = torch.fft.irfft2(
                amp * torch.exp(1j * phase_perm), s=x.shape[-2:], norm="ortho"
            ).float()
            logits_amp = self.forward_backbone(x_ifft)
            loss_hff = self.cfg.lambda_hff * (logits - logits_amp).pow(2).mean()
        else:
            loss_hff = torch.tensor(0.0, device=device)

        total_loss = loss_cls + loss_nce + loss_ace_reg + loss_hff

        return {
            "loss": total_loss,
            "logits": logits.detach(),
            "y": y.detach(),
            "ace": ace.detach(),
            "loss_components": {
                "ce": loss_cls.item(),
                "nce": loss_nce.item(),
                "ace_reg": loss_ace_reg.item(),
                "hff": loss_hff.item(),
            },
        }

# ---------------------------------------------------------------------
# Backbone builder (isolated for clarity)
# ---------------------------------------------------------------------

def build_backbone(num_classes: int = 2) -> nn.Module:
    """Create a ResNet-50 and (optionally) load DINO initialisation."""
    backbone = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)
    # Try to load DINO weights (non-fatal on failure)
    try:
        dino_state = torch.hub.load("facebookresearch/dino:main", "dino_resnet50").state_dict()
        backbone.load_state_dict(dino_state, strict=False)
        print("Loaded DINO weights for ResNet-50.")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not load DINO weights – {exc}")
    return backbone.to(device)

# ---------------------------------------------------------------------
# One-epoch training loop
# ---------------------------------------------------------------------

def _make_second_view(images: torch.Tensor, strong_aug) -> torch.Tensor:  # type: ignore
    """Generate a second heavy-augmentation view for InfoNCE.
    Images come in normalised tensor format; we convert to PIL → strong_aug → tensor.
    """
    from torchvision import transforms as T

    to_pil = T.ToPILImage()
    views: List[torch.Tensor] = []
    for img in images.cpu():
        views.append(strong_aug(to_pil(img)))
    return torch.stack(views).to(device)


def train_one_epoch(
    model: CCLiDAR,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: SimpleNamespace,
) -> float:
    """Single epoch training; returns mean loss."""
    model.train()
    running: List[float] = []

    for batch in loader:
        # second view for InfoNCE
        batch["x2"] = _make_second_view(batch["x"], cfg.strong_aug)

        with autocast(enabled=cfg.fp16):
            out = model(batch)
        scaler.scale(out["loss"]).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        running.append(out["loss"].item())

    return float(np.mean(running))
