"""
train.py
Model definitions, backbone factory and checkpoint loader.
"""
from __future__ import annotations
import os, random, math, time
from pathlib import Path
from typing import Tuple, Dict, List

# ---------------------------------------------------------------------------
# Mandatory dependencies – crash early if something is missing.
# ---------------------------------------------------------------------------
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    import torchvision, torchvision.transforms as T
except Exception as e:
    raise RuntimeError("Missing required Python libraries – aborting: " + str(e))

# Reproducibility & device ---------------------------------------------------
GLOBAL_SEED = 11
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1)  LoFT MODULES  – encoder (64-D), hyper-network, adapter wrapper
# ---------------------------------------------------------------------------
class CorruptionEncoder(nn.Module):
    """64-D hand-crafted statistic vector as described in the paper."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(24, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B × 3 × 224 × 224
        B, _, H, W = x.shape
        feats: List[torch.Tensor] = []
        # Frequency bands ---------------------------------------------------
        fft = torch.fft.rfft2(x, norm="ortho")        # (B,3,H,W/2+1)
        power = torch.abs(fft) ** 2                    # power spectrum
        bands = torch.chunk(torch.linspace(0, W//2, steps=W//2+1, device=x.device), 4)
        for b in bands:
            idx = torch.arange(power.shape[-1], device=x.device)
            mask = (idx >= b.min()) & (idx <= b.max())
            feats.append(power[..., mask].mean(dim=(-2, -1)))           # (B,3)
        # Colour stats ------------------------------------------------------
        feats += [x.mean((-2, -1)), x.flatten(2).std(-1)]               # means, stds
        # Edge density (Sobel) & blur (Laplace variance) -------------------
        sob = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device)
        sob_x = sob.view(1,1,3,3).repeat(3,1,1,1)
        sob_y = sob_x.transpose(-1,-2)
        gx = F.conv2d(x, sob_x, padding=1, groups=3)
        gy = F.conv2d(x, sob_y, padding=1, groups=3)
        feats.append(torch.sqrt(gx**2 + gy**2).mean((-2,-1)))           # edge density
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype, device=x.device)
        lap_k = lap.view(1,1,3,3).repeat(3,1,1,1)
        lap_out = F.conv2d(x, lap_k, padding=1, groups=3)
        feats.append(lap_out.var((-2,-1)))                              # blur variance
        raw = torch.cat(feats, dim=1)                                  # (B,24)
        return self.proj(torch.log1p(raw))                              # (B,64)


class HyperNetwork(nn.Module):
    """2-layer MLP that predicts γ/β per channel for four residual stages."""
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 7680)
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(z)                          # (B,7680)
        gamma, beta = out.chunk(2, dim=1)          # each (B,3840)
        return gamma, beta


class _ModBlock(nn.Module):
    """Wraps a residual stage and applies affine γ/β modulation."""
    def __init__(self, stage: nn.Sequential, n_ch: int):
        super().__init__()
        self.stage = stage
        self.n_ch = n_ch

    def forward(self, x, g, b):
        y = self.stage(x)
        return g.view(-1, self.n_ch, 1, 1) * y + b.view(-1, self.n_ch, 1, 1)


class LoFTAdapter(nn.Module):
    """Adapter that grafts LoFT onto a torchvision ResNet-50 backbone."""
    CH = [256, 512, 1024, 2048]

    def __init__(self, backbone: torchvision.models.ResNet):
        super().__init__()
        self.enc = CorruptionEncoder()
        self.hnet = HyperNetwork()
        # Backbone graft ----------------------------------------------------
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = _ModBlock(backbone.layer1, self.CH[0])
        self.layer2 = _ModBlock(backbone.layer2, self.CH[1])
        self.layer3 = _ModBlock(backbone.layer3, self.CH[2])
        self.layer4 = _ModBlock(backbone.layer4, self.CH[3])
        self.avgpool, self.fc = backbone.avgpool, backbone.fc

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        g, b = self.hnet(z)                      # (B,3840) each
        # Split per stage ---------------------------------------------------
        splits = torch.split(torch.arange(3840), self.CH, dim=0)
        g_s = [g[:, s] for s in splits]
        b_s = [b[:, s] for s in splits]
        x = self.stem(x)
        x = self.layer1(x, g_s[0], b_s[0])
        x = self.layer2(x, g_s[1], b_s[1])
        x = self.layer3(x, g_s[2], b_s[2])
        x = self.layer4(x, g_s[3], b_s[3])
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ---------------------------------------------------------------------------
# 2)  Backbone factory & checkpoint loader
# ---------------------------------------------------------------------------
BACKBONE_FACTORY = {
    "resnet50": lambda: torchvision.models.resnet50(weights=None),
}

VARIANTS = ["vanilla", "augmix", "bn_adapt", "damp", "stylenorm", "loft"]
weights_root = Path("weights")

def load_model(arch: str, variant: str, seed: int):
    """Instantiate model architecture and load the corresponding checkpoint."""
    fname = f"{arch}_{variant}_s{seed}.pt"
    ckpt_path = weights_root / fname
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found – did you download weights?")
    model = BACKBONE_FACTORY[arch]()
    if variant == "loft":
        model = LoFTAdapter(model)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()

# ---------------------------------------------------------------------------
# 3)  Quick implementation unit test (executed only when run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running LoFT unit-tests …", flush=True)
    _test_img = torch.rand(2,3,224,224)
    assert CorruptionEncoder()(_test_img).shape == (2,64), "Encoder output must be 64-D"
    base = torchvision.models.resnet50(weights=None)
    loft = LoFTAdapter(base)
    params_ratio = sum(p.numel() for p in loft.parameters()) / sum(p.numel() for p in base.parameters())
    assert params_ratio < 1.01, "LoFT adds more than 1 % parameters"
    with torch.no_grad():
        out = loft(_test_img)
    assert out.shape == (2,1000), "Forward pass failed"
    print(f"✓ LoFT implementation verified.  Extra params: {(params_ratio-1)*100:.2f} %")
