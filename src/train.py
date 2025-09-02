from __future__ import annotations
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils import add_self_loops

###############################################################################
#                                1.  Utilities                                #
###############################################################################

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Node classification accuracy in %."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item() * 100.0


def apsd(x: torch.Tensor) -> float:
    """Average pair-wise squared distance on a (sub-)sample of nodes."""
    if x.size(0) > 4096:
        idx = torch.randperm(x.size(0), device=x.device)[:4096]
        x = x[idx]
    return (torch.pdist(x, p=2.0) ** 2).mean().item()

###############################################################################
#                          2.  ADR-GNN Building Blocks                        #
###############################################################################

class ADRConv(nn.Module):
    """Adaptive Reaction–Diffusion layer.

    H_{l+1} = LN( (1 + γθ_i) H_l  +  η P H_l ) W_l
    where θ_i is a learnable node-wise gate produced by a small MLP.
    """

    def __init__(self, in_dim: int, eta_init: float = 0.9, gamma_init: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim, bias=False)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.ln = nn.LayerNorm(in_dim)

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        edge_index: torch.Tensor,
        deg_norm: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index  # COO
        diffused = deg_norm[row].unsqueeze(1) * x[col]
        diffused = torch.zeros_like(x).scatter_add_(
            0, row.unsqueeze(1).repeat(1, x.size(1)), diffused
        )

        gate_in = torch.cat([x0, x, deg_norm.unsqueeze(1)], dim=1)
        theta = self.mlp(gate_in).squeeze()  # (N,)
        reaction = (1.0 + self.gamma * theta).unsqueeze(1) * x
        out = reaction + self.eta * diffused
        out = self.proj(out)
        return self.ln(out)

###############################################################################
#                               3.  GNN Models                                #
###############################################################################

class ADRGNN(nn.Module):
    """Our proposed ADR-GNN stack."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, depth: int, dropout: float = 0.6):
        super().__init__()
        self.x0: torch.Tensor | None = None
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([ADRConv(hidden) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg_norm: torch.Tensor):
        if self.x0 is None or self.x0.size(0) != x.size(0):
            self.x0 = x.clone().detach()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_proj(x))
        x0 = x.clone().detach()
        for layer in self.layers:
            x = layer(x, x0, edge_index, deg_norm)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_proj(x)


class VanillaGCN(nn.Module):
    """PairNorm-regularised vanilla GCN used as baseline."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, depth: int = 2, dropout: float = 0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        if depth == 1:
            self.convs.append(GCNConv(in_dim, num_classes))
        else:
            self.convs.append(GCNConv(in_dim, hidden))
            for _ in range(depth - 2):
                self.convs.append(GCNConv(hidden, hidden))
            self.convs.append(GCNConv(hidden, num_classes))
        self.pairnorm = PairNorm()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, *_):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = self.pairnorm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

###############################################################################
#                          4.  Training / Evaluation                          #
###############################################################################

def train_model(
    model: nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    criterion,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int,
    patience: int,
) -> Dict:
    """Generic training loop with early stopping."""
    best_val = -1.0
    best_state = None
    history: Dict[str, list] = {"train_acc": [], "val_acc": [], "apsd": []}
    wait = 0

    for epoch in range(1, epochs + 1):
        # -------------------- train -------------------- #
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.deg_norm)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # -------------------- eval --------------------- #
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.deg_norm)
            tr = accuracy(logits[train_mask], data.y[train_mask])
            va = accuracy(logits[val_mask], data.y[val_mask])
            history["train_acc"].append(tr)
            history["val_acc"].append(va)
            if epoch % 5 == 0:
                history["apsd"].append(apsd(logits))

        # ----------------- early stop ------------------ #
        if va > best_val:
            best_val = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed to improve")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.deg_norm)
        history["test_acc"] = accuracy(logits[test_mask], data.y[test_mask])
    return history
