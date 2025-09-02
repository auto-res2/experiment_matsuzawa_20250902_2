"""src/train.py – model architectures and training utilities for the ACuDiN project
The code is extracted and refactored from the original monolithic experimental
script.  It contains:
• Device / AMP configuration that is safe for both GPU and CPU environments.
• Model building blocks (EdgeMLP, NodeMLP, ACuDiN, baselines).
• Training helpers (EarlyStop, train_epoch).
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, APPNP, BatchNorm, PairNorm, MessagePassing
)
from torch_geometric.utils import degree

# -----------------------------------------------------------------------------
# Device / precision setup -----------------------------------------------------
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32  # safer on CPU
    print("[WARN] CUDA not available – running on CPU. Training will be slow.")

AMP_SCALER = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

# -----------------------------------------------------------------------------
# Model components -------------------------------------------------------------
# -----------------------------------------------------------------------------
class EdgeMLP(nn.Module):
    """Small MLP that predicts the edge gate ϕ_ij ∈ [0,1]."""

    def __init__(self, in_dim: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.PReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor):  # type: ignore
        return self.net(z).view(-1)


class NodeMLP(nn.Module):
    """Small MLP that predicts the node specific teleport probability α_i."""

    def __init__(self, in_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.PReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor):  # type: ignore
        return self.net(z).view(-1)


class ACuDiNLayer(MessagePassing):
    """One layer of Adaptive Curvature-guided Diffusion Network (ACuDiN)."""

    def __init__(self, in_dim: int, out_dim: int, final: bool = False):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.final = final
        if not final:
            self.bn = BatchNorm(out_dim)
            self.act = nn.PReLU()
        self.edge_mlp = EdgeMLP()
        self.alpha_mlp = NodeMLP(in_dim + 2)  # features + degree + variance

    # ----------------------------------
    @staticmethod
    def approx_curvature(edge_index: torch.Tensor, deg: torch.Tensor):
        di = deg[edge_index[0]]
        dj = deg[edge_index[1]]
        return (1.0 / (di + 1e-6) + 1.0 / (dj + 1e-6)).unsqueeze(-1)

    # ----------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # type: ignore
        # ϕ_ij  --------------------------------------------------------------
        with torch.no_grad():
            deg = degree(edge_index[0], num_nodes=x.size(0), dtype=x.dtype, device=x.device)
            kappa = self.approx_curvature(edge_index, deg)
            edge_feat = torch.cat(
                [
                    kappa,
                    deg[edge_index[0]].unsqueeze(-1),
                    deg[edge_index[1]].unsqueeze(-1),
                    torch.ones_like(kappa),  # bias term
                ],
                dim=-1,
            )
        phi = self.edge_mlp(edge_feat)

        # Message passing ---------------------------------------------------
        x_msg = self.propagate(edge_index, x=x, phi=phi)
        out = self.lin(x_msg)

        if self.final:
            return out

        # α_i  --------------------------------------------------------------
        feat_var = x.var(dim=-1, keepdim=True, unbiased=False)
        node_feat = torch.cat([x, deg.unsqueeze(-1), feat_var], dim=-1)
        alpha = self.alpha_mlp(node_feat).unsqueeze(-1)
        out = (1 - alpha) * out + alpha * x  # APPNP-style blend
        out = self.bn(out)
        return self.act(out)

    # ------------------------------------------------------------------
    def message(self, x_j: torch.Tensor, phi: torch.Tensor):  # type: ignore
        return phi.unsqueeze(-1) * x_j


class ACuDiN(nn.Module):
    """Stack of ACuDiN layers with dropout between consecutive layers."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int):
        super().__init__()
        assert layers >= 2, "Need at least 2 layers"
        mods = [ACuDiNLayer(in_dim, hidden)]
        for _ in range(layers - 2):
            mods.append(ACuDiNLayer(hidden, hidden))
        mods.append(ACuDiNLayer(hidden, out_dim, final=True))
        self.layers = nn.ModuleList(mods)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # type: ignore
        h = x
        for layer in self.layers[:-1]:
            h = self.dropout(layer(h, edge_index))
        return self.layers[-1](h, edge_index)


# -----------------------------------------------------------------------------
# Baseline models -------------------------------------------------------------
# -----------------------------------------------------------------------------
class StackedGCN(nn.Module):
    """Vanilla GCN with optional deep stacking and batch norms."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        assert layers >= 2, "Need ≥2 layers"
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)])
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_dim))
        self.bns = nn.ModuleList([BatchNorm(hidden) for _ in range(layers - 1)])
        self.drop = nn.Dropout(0.5)
        # PReLU slope is learned per layer for flexibility
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # type: ignore
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            h = self.bns[i](h)
            h = self.prelu(h)
            h = self.drop(h)
        return self.convs[-1](h, edge_index)


# -----------------------------------------------------------------------------
# Model builder ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_model(name: str, in_dim: int, out_dim: int, hidden: int, layers: int):
    """Factory that returns the requested model."""

    name = name.lower()
    if name == "gcn":
        return StackedGCN(in_dim, out_dim, hidden, layers)
    if name == "appnp":
        return APPNP(
            K=10,
            alpha=0.1,
            dropout=0.5,
            cached=False,
            add_self_loops=True,
            in_channels=in_dim,
            out_channels=out_dim,
        )

    if name == "pairnorm":
        class PairNormGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList(
                    [GCNConv(in_dim, hidden, add_self_loops=False)]
                )
                for _ in range(layers - 2):
                    self.convs.append(GCNConv(hidden, hidden, add_self_loops=False))
                self.convs.append(GCNConv(hidden, out_dim, add_self_loops=False))
                self.pn = PairNorm()

            def forward(self, x, edge_index):  # type: ignore
                h = x
                for conv in self.convs[:-1]:
                    h = F.relu(conv(h, edge_index))
                    h = self.pn(h)
                return self.convs[-1](h, edge_index)

        return PairNormGCN()

    if name == "dropedge":
        class DropEdgeResGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList([GCNConv(in_dim, hidden)])
                for _ in range(layers - 2):
                    self.convs.append(GCNConv(hidden, hidden))
                self.convs.append(GCNConv(hidden, out_dim))

            def forward(self, x, edge_index):  # type: ignore
                h = x
                for i, conv in enumerate(self.convs[:-1]):
                    ei = edge_index
                    # DropEdge with p = 0.2 during training only
                    if self.training and torch.rand(()) < 0.2:
                        mask = torch.rand(ei.size(1), device=ei.device) > 0.2
                        ei = ei[:, mask]
                    h_new = F.relu(conv(h, ei))
                    h = h + h_new if i % 2 == 1 else h_new  # residual every 2 layers
                return self.convs[-1](h, edge_index)

        return DropEdgeResGCN()

    if name == "acudin":
        return ACuDiN(in_dim, hidden, out_dim, layers)

    raise ValueError(f"Unknown model '{name}'")


# -----------------------------------------------------------------------------
# Training helpers -------------------------------------------------------------
# -----------------------------------------------------------------------------
class EarlyStop:
    """Simple validation-based early stopper."""

    def __init__(self, patience: int):
        self.best = -float("inf")
        self.bad = 0
        self.patience = patience

    def step(self, val: float) -> bool:
        if val > self.best:
            self.best = val
            self.bad = 0
            return False
        self.bad += 1
        return self.bad > self.patience


# ----------------------------------

def train_epoch(model: nn.Module, data, mask: torch.Tensor, optimiser):
    """One optimisation step with AMP support on GPU and a safe CPU fallback."""

    import torch.utils.data  # local import to keep public surface tiny

    model.train()
    optimiser.zero_grad(set_to_none=True)

    autocast_ctx = (
        torch.cuda.amp.autocast(device_type="cuda", dtype=DTYPE)
        if DEVICE.type == "cuda"
        else contextlib.nullcontext()
    )

    with autocast_ctx:
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[mask], data.y[mask])

    AMP_SCALER.scale(loss).backward()
    AMP_SCALER.step(optimiser)
    AMP_SCALER.update()
    return loss.item()


# -----------------------------------------------------------------------------
# What to export when doing ``from src.train import *`` ------------------------
# -----------------------------------------------------------------------------
__all__ = [
    "DEVICE",
    "DTYPE",
    "AMP_SCALER",
    "EarlyStop",
    "train_epoch",
    "build_model",
]
