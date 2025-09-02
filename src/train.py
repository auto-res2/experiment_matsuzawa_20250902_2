"""train.py
LoFT modules: corruption encoder, hyper-network and adapter that wraps a
ResNet-style backbone.  No training loop is implemented because the
original script performs only inference on synthetic data.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ----------------------------------------------------------------------------------
# 1)  Corruption-encoder (fast, physically interpretable statistics)
# ----------------------------------------------------------------------------------
class CorruptionEncoder(nn.Module):
    """Compute a 36-D embedding composed of frequency-domain power, colour
    moments, edge density and blur variance.  The implementation is identical
    to the monolithic experimental script but condensed into a reusable module.
    """

    def __init__(self) -> None:
        super().__init__()
        # keeps the module from being treated as "empty" when tracing / scripting
        self.register_buffer("dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,224,224)
        B = x.size(0)
        feats: List[torch.Tensor] = []

        # 1. Log-spectral power in four frequency bands (width axis)
        fft = torch.fft.rfft2(x, norm="ortho")  # (B,3,H,W/2+1)
        power = torch.abs(fft) ** 2
        freq_bins = torch.linspace(0, 1, steps=power.shape[-1], device=x.device)
        bands = torch.chunk(freq_bins, 4)
        for band in bands:
            idx = (freq_bins >= band.min()) & (freq_bins <= band.max())
            feats.append(power[..., idx].mean(dim=(-2, -1)))  # (B,3)

        # 2. Per-channel first & second moments (mean / std)
        mean = x.mean(dim=(-2, -1))                # (B,3)
        std  = x.flatten(2).std(dim=-1)            # (B,3)
        feats.extend([mean, std])

        # 3. Edge density (Sobel) & blur variance (Laplacian)
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        sobel_x = sobel_x.repeat(3, 1, 1, 1)
        sobel_y = sobel_y.repeat(3, 1, 1, 1)
        gx = F.conv2d(x, sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, sobel_y, padding=1, groups=3)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2)
        edge_density = grad_mag.mean(dim=(-2, -1))
        feats.append(edge_density)

        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        lap_kernel = lap_kernel.repeat(3, 1, 1, 1)
        lap = F.conv2d(x, lap_kernel, padding=1, groups=3)
        blur_var = lap.var(dim=(-2, -1))
        feats.append(blur_var)

        feat = torch.cat(feats, dim=1)   # (B,36)
        proj = torch.log1p(feat)         # (B,36) – log-scale for stability
        return proj


# ----------------------------------------------------------------------------------
# 2)  Hyper-network: predicts affine γ,β scalars for four stages
# ----------------------------------------------------------------------------------
class HyperNetwork(nn.Module):
    def __init__(self, in_dim: int = 36, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 8)
        )

    def forward(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(emb)          # (B,8)
        gamma, beta = out.chunk(2, dim=1)
        return gamma, beta           # each (B,4)


# ----------------------------------------------------------------------------------
# 3)  Stage-modulator: broadcast γ,β across feature maps of a residual stage
# ----------------------------------------------------------------------------------
class _StageModulator(nn.Module):
    def __init__(self, stage: nn.Module) -> None:
        super().__init__()
        self.stage = stage

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        y = self.stage(x)
        y = (gamma.view(-1, 1, 1, 1) * y) + beta.view(-1, 1, 1, 1)
        return y


# ----------------------------------------------------------------------------------
# 4)  LoFT wrapper that plugs the components into a ResNet-style backbone
# ----------------------------------------------------------------------------------
class LoFTAdapter(nn.Module):
    """Wrap a torchvision ResNet backbone with LoFT conditioning.  Only ResNet
    family is supported in this demo implementation because the original script
    used ResNet-50 exclusively.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        if not hasattr(backbone, "layer1"):
            raise ValueError("LoFTAdapter currently supports ResNet-style backbones only.")

        self.encoder = CorruptionEncoder()
        self.hyper   = HyperNetwork(in_dim=36, hidden=128)

        # Decompose backbone into stem / four stages / head
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = _StageModulator(backbone.layer1)
        self.layer2 = _StageModulator(backbone.layer2)
        self.layer3 = _StageModulator(backbone.layer3)
        self.layer4 = _StageModulator(backbone.layer4)
        self.avgpool = backbone.avgpool
        self.fc      = backbone.fc

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)               # (B,36)
        gamma, beta = self.hyper(emb)       # each (B,4)
        g1, g2, g3, g4 = gamma.unbind(1)    # (B,)
        b1, b2, b3, b4 = beta.unbind(1)

        x = self.stem(x)
        x = self.layer1(x, g1, b1)
        x = self.layer2(x, g2, b2)
        x = self.layer3(x, g3, b3)
        x = self.layer4(x, g4, b4)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
