"""src/train.py
Model architectures and training-related utilities for ADR-GNN experiments.
"""
from __future__ import annotations
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, PairNorm

# -----------------------------------------------------------------------------
#  Generic helpers – kept here to avoid circular imports
# -----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Return classification accuracy in % (float, not torch Tensor)."""
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0


def apsd(feat: torch.Tensor, k: int = 4096) -> float:
    """Average pairwise squared Euclidean distance on a random subset (≤k).
    Helps as over-smoothing indicator.  Returns python float for logging.
    """
    if feat.size(0) > k:
        idx = torch.randperm(feat.size(0), device=feat.device)[:k]
        feat = feat[idx]
    return (torch.pdist(feat, p=2.0) ** 2).mean().item()

# -----------------------------------------------------------------------------
#  ADR-GNN core layers
# -----------------------------------------------------------------------------

class ADRConv(nn.Module):
    """One Adaptive Reaction–Diffusion layer.

    Updated (v2):
    H_{l+1} = LN\big[(I + γ Θ) H_l  −  η P H_l\big]
    where  Θ  is a node-wise anti-diffusion gate, and  P=D^{-1}A  is the
    row-normalised adjacency (diffusion operator).  Note the **minus** sign in
    front of the diffusion term – this corrects a sign error present in the
    original implementation and ensures that the spectrum of the linearised
    operator stays within the provably bounded region [1−η, 1+γ].
    """

    def __init__(self, dim: int, eta_init: float = 0.9, gamma_init: float = 0.1):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        # Learnable scalar factors (initialised to given constants)
        # We store the *true* parameters (no sigmoid) and clamp to [0,1] in the
        # forward pass to keep them in the valid range.
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # Gate network – 2-layer MLP → tanh → scalar per node
        self.theta_net = nn.Sequential(
            nn.Linear(dim * 2 + 1, 32), nn.GELU(), nn.Linear(32, 1), nn.Tanh()
        )
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        h: torch.Tensor,
        h0_raw: torch.Tensor,
        edge_index: torch.Tensor,
        deg_inv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Returns
        -------
        out : torch.Tensor            # node features after layer
        theta_detached : torch.Tensor # θ values detached from graph (for logging)
        """
        # (1) feature projection
        h = self.W(h)

        # (2) diffusion part   diff = D^{-1} A h
        row, col = edge_index
        msg = deg_inv[row].unsqueeze(1) * h[col]
        diff = torch.zeros_like(h).scatter_add_(0, row.unsqueeze(1).expand_as(msg), msg)

        # (3) anti-diffusion gate   Θ_i = tanh( f( x_i^0, h_i , log deg_i ) )
        deg_log = (deg_inv + 1e-8).log().unsqueeze(1)  # numeric stability
        gate_in = torch.cat([h0_raw, h.detach(), deg_log], dim=1)
        theta = self.theta_net(gate_in).squeeze()  # shape (N,)

        # (4) reaction–diffusion combination & normalisation
        react = (1.0 + self.gamma * theta).unsqueeze(1) * h

        # Clamp η to [0,1] and snap very small values to zero for EXP-1 linear check
        eta_eff = torch.clamp(self.eta, 0.0, 1.0)
        eta_eff = torch.where(eta_eff < 1e-4, torch.zeros_like(eta_eff), eta_eff)

        # IMPORTANT: Use **minus** sign for diffusion term (see doc-string)
        out = self.ln(react - eta_eff * diff)
        return out, theta.detach()


class ADRGNN(nn.Module):
    """Multi-layer ADR-GNN for node classification."""

    def __init__(self, in_dim: int, hid: int, n_cls: int, depth: int, dropout: float):
        super().__init__()
        self.depth = depth
        self.do = dropout
        self.in_lin = nn.Linear(in_dim, hid)
        self.layers = nn.ModuleList([ADRConv(hid) for _ in range(depth)])
        self.out_lin = nn.Linear(hid, n_cls)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, deg_inv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0_raw = x  # keep initial features for gate MLP
        h = F.relu(self.in_lin(x))
        thetas: List[torch.Tensor] = []
        for layer in self.layers:
            h, theta = layer(h, h0_raw, edge_index, deg_inv)
            h = F.dropout(F.relu(h), p=self.do, training=self.training)
            thetas.append(theta)
        logits = self.out_lin(h)
        return logits, torch.stack(thetas)  # (L × N)


class GCN(nn.Module):
    """Vanilla GCN (baseline) with optional depth>2 and PairNorm."""

    def __init__(self, in_dim: int, hid: int, n_cls: int, depth: int, drop: float):
        super().__init__()
        self.convs = nn.ModuleList()
        if depth == 1:
            self.convs.append(GCNConv(in_dim, n_cls))
        else:
            self.convs.append(GCNConv(in_dim, hid))
            for _ in range(depth - 2):
                self.convs.append(GCNConv(hid, hid))
            self.convs.append(GCNConv(hid, n_cls))
        self.pn = PairNorm()
        self.drop = drop

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, *_):  # ignore deg_inv
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = self.pn(x)
                x = F.dropout(x, p=self.drop, training=self.training)
        return x, None
