"""src/train.py
Core model architectures and one training step helper.
Refactored from the original monolithic script.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, APPNP, BatchNorm
from torch_geometric.utils import degree

# -----------------------------------------------------------------------------
#  Global constants & fail-fast guards
# -----------------------------------------------------------------------------
assert torch.__version__.startswith("2.2"), "PyTorch 2.2.* is required"
assert torch.cuda.is_available(), "CUDA GPU required for the experiments"

DEVICE = torch.device("cuda")
DTYPE = torch.float16  # used for torch.autocast
AMP_SCALER = torch.cuda.amp.GradScaler()

# -----------------------------------------------------------------------------
#  ACuDiN building blocks
# -----------------------------------------------------------------------------
class EdgeMLP(nn.Module):
    """ϕ_ij gate – 2-layer MLP mapping local curvature & degree to [0,1]."""

    def __init__(self, in_dim: int = 4, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.PReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, edge_feats: torch.Tensor):
        return self.mlp(edge_feats).view(-1)


class NodeMLP(nn.Module):
    """α_i gate – 2-layer MLP mapping node feature statistics to (0,1)."""

    def __init__(self, feat_dim: int, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.PReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor):
        return self.mlp(h).view(-1)


class ACuDiNLayer(nn.Module):
    """One ACuDiN layer: curvature-gated message passing + node-wise diffusion."""

    def __init__(self, in_dim: int, out_dim: int, final: bool = False):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=False)
        self.edge_mlp = EdgeMLP()
        self.bn = BatchNorm(out_dim, affine=True, track_running_stats=True)
        self.final = final
        if not final:
            self.act = nn.PReLU()
        # +2 stats: degree & feature variance
        self.alpha_mlp = NodeMLP(out_dim if final else in_dim + 2)
        # When feature dimensions differ we need a projection for the residual/teleport path
        self.skip_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    # ---------------------------------------------------------------------
    @staticmethod
    def approx_curvature(edge_index: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Cheap O(|E|) Ollivier–Ricci curvature proxy κ_ij ≈ 1/deg_i + 1/deg_j."""
        di = deg[edge_index[0]]
        dj = deg[edge_index[1]]
        kappa = 1.0 / di + 1.0 / dj  # proxy in (0,2]
        return kappa.unsqueeze(-1)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor):
        # Edge-wise gate ϕ
        kappa = self.approx_curvature(edge_index, deg).to(x.dtype)
        deg_i = deg[edge_index[0]].unsqueeze(-1)
        deg_j = deg[edge_index[1]].unsqueeze(-1)
        edge_feat = torch.cat([kappa, deg_i, deg_j, kappa * 0 + 1], dim=-1)
        phi = self.edge_mlp(edge_feat)  # (E,)

        out = self.conv(x, edge_index, edge_weight=phi)

        # Node-wise teleport α
        if self.final:
            return out
        feat_var = (x.var(dim=-1, unbiased=False, keepdim=True) + 1e-6)
        node_feat = torch.cat([x, deg.unsqueeze(-1), feat_var], dim=-1)
        alpha = self.alpha_mlp(node_feat).unsqueeze(-1)  # (N,1)
        # Project x to match out_dim when necessary for the skip/teleport connection
        x_proj = self.skip_proj(x)
        out = (1 - alpha) * out + alpha * x_proj
        out = self.bn(out)
        return self.act(out)


class ACuDiN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = [ACuDiNLayer(in_dim, hidden)]
        for _ in range(num_layers - 2):
            layers.append(ACuDiNLayer(hidden, hidden))
        layers.append(ACuDiNLayer(hidden, out_dim, final=True))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=0.5)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        deg = degree(edge_index[0], num_nodes=x.size(0)).to(x.device)
        h = x
        for layer in self.layers[:-1]:
            h = self.dropout(layer(h, edge_index, deg))
        logits = self.layers[-1](h, edge_index, deg)
        return logits


# -----------------------------------------------------------------------------
#  Baseline factory
# -----------------------------------------------------------------------------

def make_baseline(name: str, in_dim: int, out_dim: int, num_layers: int, hidden: int = 64):
    """Return a simple baseline model given its name."""
    name = name.lower()

    if name == "gcn":

        class StackedGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(GCNConv(in_dim, hidden))
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden, hidden))
                self.convs.append(GCNConv(hidden, out_dim))
                self.bn = nn.ModuleList([BatchNorm(hidden) for _ in range(num_layers - 1)])
                self.dropout = nn.Dropout(0.5)

            def forward(self, x, edge_index):
                h = x
                for i, conv in enumerate(self.convs[:-1]):
                    h = conv(h, edge_index)
                    h = self.bn[i](h)
                    h = F.prelu(h, torch.tensor(0.25, device=h.device))
                    h = self.dropout(h)
                return self.convs[-1](h, edge_index)

        return StackedGCN()

    elif name == "appnp":
        # returning PyG's built-in APPNP
        return APPNP(
            K=10,
            alpha=0.1,
            dropout=0.5,
            cached=False,
            add_self_loops=True,
            in_channels=in_dim,
            out_channels=out_dim,
        )

    raise NotImplementedError(f"Baseline {name} not supported.")


# -----------------------------------------------------------------------------
#  Single training step helper
# -----------------------------------------------------------------------------

def train_step(model: nn.Module, data, optimizer: torch.optim.Optimizer, mask: torch.Tensor) -> float:
    """Perform one optimisation step and return the loss as Python float."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=DTYPE):
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])

    AMP_SCALER.scale(loss).backward()
    AMP_SCALER.step(optimizer)
    AMP_SCALER.update()
    return loss.item()


# -----------------------------------------------------------------------------
#  Misc helpers
# -----------------------------------------------------------------------------

def ds_num_classes(data) -> int:
    """Utility: infer number of classes from a PyG data object."""
    return int(data.y.max().item() + 1)
