"""src/train.py
Core training-related components: model architectures, losses, trainer
---------------------------------------------------------------------
Author: Cutting-edge AI Researcher
Licence: MIT
"""
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --------------------------------------------------------------------
#  Global constants & reproducibility helpers
# --------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
DATA_DIR = Path("data")
FIG_DIR = Path("figures")
CKPT_DIR = Path("checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Ensure experiment-level determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------
#  Counterfactual generator utilities (used by CRAFT, not baseline ERM)
# --------------------------------------------------------------------

def init_inpaint_pipe() -> StableDiffusionInpaintPipeline:
    """Initialise Stable-Diffusion in-painting pipeline on the chosen DEVICE."""
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


def generate_counterfactuals(
    pipe: StableDiffusionInpaintPipeline,
    images: torch.Tensor,
    prompts: List[str],
    M: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate `M` in-painted counterfactuals per image.

    Returns (cf_imgs, masks) where both have shape (B, M, C, H, W).
    """
    B, C, H, W = images.shape
    cf_imgs = torch.zeros((B, M, C, H, W), dtype=torch.float16, device=images.device)
    masks = torch.zeros_like(cf_imgs)

    to_pil = transforms.ToPILImage()
    for b in range(B):
        image_pil = to_pil(images[b].cpu())
        for m in range(M):
            # random square region to in-paint (48–96 px)
            sz = random.randint(48, 96)
            x0 = random.randint(0, H - sz)
            y0 = random.randint(0, W - sz)

            mask_img = Image.new("RGB", (W, H), (0, 0, 0))
            for i in range(x0, x0 + sz):
                for j in range(y0, y0 + sz):
                    mask_img.putpixel((j, i), (255, 255, 255))

            pipe_out = pipe(prompt=prompts[b], image=image_pil, mask_image=mask_img)
            img_cf = pipe_out.images[0]
            cf_imgs[b, m] = transforms.ToTensor()(img_cf).to(images.device)
            masks[b, m, :, x0 : x0 + sz, y0 : y0 + sz] = 1
    return cf_imgs, masks

# --------------------------------------------------------------------
#  Model definitions
# --------------------------------------------------------------------


class DualHead(nn.Module):
    """Dual-head classifier used by CRAFT (causal & context heads)."""

    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True)

        # Handle both ResNet & ViT style classifiers
        if hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "head"):
            in_features = self.backbone.head.in_features
            self.backbone.reset_classifier(0)
        else:
            raise ValueError("Unknown backbone structure for model: %s" % backbone_name)

        self.h_causal = nn.Sequential(
            nn.Linear(in_features, num_classes), nn.BatchNorm1d(num_classes)
        )
        self.h_context = nn.Sequential(
            nn.Linear(in_features, num_classes), nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        z = self.backbone(x)
        return self.h_causal(z) + self.h_context(z), z


class SingleHead(nn.Module):
    """Ablation model with a single classifier head (used in Experiment 3)."""

    def __init__(self, backbone_name: str = "resnet50", num_classes: int = 2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        in_features = self.backbone.fc.in_features  # ResNet-specific; safe for RN50 here
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):  # keep signature consistent with DualHead
        return self.backbone(x), None


# --------------------------------------------------------------------
#  Losses
# --------------------------------------------------------------------

def craft_loss(
    backbone: nn.Module,
    h_causal: nn.Module,
    cf_imgs: torch.Tensor,
    orig_preds: torch.Tensor,
    lambda_: float = 1.0,
) -> torch.Tensor:
    """Squared-difference causal regulariser per CRAFT paper."""

    B, M, C, H, W = cf_imgs.shape
    cf_flat = cf_imgs.view(B * M, C, H, W)

    with torch.no_grad():
        cf_z = backbone(cf_flat)
    cf_z = cf_z.view(B, M, -1)
    preds_cf = h_causal(cf_z)  # (B, M, num_cls)

    orig = orig_preds.unsqueeze(1).detach()  # (B, 1, num_cls)
    diff = (preds_cf - orig) ** 2
    return lambda_ * diff.mean()

# --------------------------------------------------------------------
#  Trainer wrapper
# --------------------------------------------------------------------

class Trainer:
    """Light-weight training loop wrapper (AdamW + cosine schedule)."""

    def __init__(
        self,
        model: nn.Module,
        loader_train: DataLoader,
        loader_val: DataLoader,
        num_epochs: int,
        experiment_name: str,
        lr: float = 3e-4,
        weight_decay: float = 5e-2,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = loader_train
        self.val_loader = loader_val
        self.num_epochs = num_epochs
        self.name = experiment_name

        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=num_epochs
        )

        self.best_acc: float = 0.0
        self.best_state: Dict[str, Any] | None = None

    # ----------------------- training helpers ----------------------- #
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, n_samples = 0.0, 0
        for x, y, _ in tqdm(self.train_loader, desc=f"train e{epoch}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds, _ = self.model(x)
            loss = F.cross_entropy(preds, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        return total_loss / n_samples

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds, _ = self.model(x)
            correct += preds.argmax(1).eq(y).sum().item()
            total += x.size(0)
        return 100.0 * correct / total

    # ----------------------------- run ------------------------------ #
    def run(self):
        for epoch in range(self.num_epochs):
            t0 = time.time()
            loss_epoch = self.train_one_epoch(epoch)
            acc_val = self.evaluate(self.val_loader)
            self.scheduler.step()

            if acc_val > self.best_acc:
                self.best_acc = acc_val
                # store on CPU to avoid GPU memory leak
                self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

            dur_min = (time.time() - t0) / 60.0
            print(
                f"[Epoch {epoch+1}/{self.num_epochs}] loss={loss_epoch:.4f}  val_acc={acc_val:.2f}%  time={dur_min:.1f}min"
            )

        # restore best chkpt
        assert self.best_state is not None, "Training finished but best_state is None!"
        self.model.load_state_dict(self.best_state)
