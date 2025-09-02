"""src/evaluate.py
Experiment routines (Exp-1/2/3) and evaluation-related utilities.
"""
from __future__ import annotations
import os, json, math, random, argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .train import ADRConv, ADRGNN, accuracy, apsd
from .preprocess import (
    ROOT, DATA_DIR, FIG_DIR, DEVICE, QUICK_MODE, set_seed, download,
)

# fvcore is optional – only used for FLOP analysis; ignore if not available
try:
    from fvcore.nn import FlopCountAnalysis  # noqa: F401  (import needed for exp-3)
    FVCORE_OK = True
except ImportError:
    FVCORE_OK = False

SEED_LIST = [0, 1] if QUICK_MODE else [10, 11, 12, 13, 14]

# -----------------------------------------------------------------------------
#  EXPERIMENT 1 – implementation & theory match
# -----------------------------------------------------------------------------

def exp1() -> None:
    print("\n================ EXPERIMENT 1 – IMPLEMENTATION & THEORY-MATCH ================")
    print("Running unit-style assertions to guarantee that ADR-GNN matches the "
          "mathematical operator and spectral bounds.\n")

    # (1) tiny ring graph (4 nodes)
    A = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    edge_index = torch.nonzero(A, as_tuple=False).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=4)  # ensures 1 self-loop / node
    deg = torch.bincount(edge_index[0], minlength=4).float()
    deg_inv = 1.0 / (deg + 1e-8)
    x = torch.randn(4, 8)

    # T1: linear equivalence when η≈0 & γ=0
    lin_layer = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False), torch.nn.LayerNorm(8))
    adr1 = ADRConv(8)
    with torch.no_grad():
        adr1.W.weight.copy_(lin_layer[0].weight)
        adr1.eta.data.fill_(-10.0)  # sigmoid→0
        adr1.gamma.data.fill_(0.0)
    out_adr, _ = adr1(x, x, edge_index, deg_inv)
    out_lin = lin_layer(x)
    diff = (out_adr - out_lin).norm().item()
    assert diff < 1e-6, f"Linear-equivalence test failed (diff {diff})"

    # T2: spectrum bound with random Θ sample
    eta, gamma = 0.9, 0.1
    adr2 = ADRConv(8, eta, gamma)
    with torch.no_grad():
        adr2.W.weight.copy_(torch.eye(8))  # identity W for clarity
    theta = adr2.theta_net(torch.cat([x, x, deg_inv.log().unsqueeze(1)], dim=1)).squeeze()

    # Build diffusion operator P **including the self-loops** so that its spectrum
    # lies within [0, 1] – this matches the operator used inside ADRConv.
    P = torch.zeros(4, 4, dtype=x.dtype)
    P[edge_index[0], edge_index[1]] = deg_inv[edge_index[0]]

    J = torch.eye(4) + gamma * torch.diag(theta) - eta * P
    eig = torch.linalg.eigvals(J).real.detach()  # detach so subsequent .numpy() calls work
    assert (eig <= 1 + gamma + 1e-4).all() and (eig >= 1 - eta - 1e-4).all(), "Spectrum bound violated"

    # T3: APSD monotonicity on 64-layer ER graph (200 nodes)
    G = nx.erdos_renyi_graph(200, 0.04, seed=0, directed=False)
    ei = torch.tensor(list(G.edges())).t().contiguous()
    ei, _ = add_self_loops(ei, num_nodes=200)
    deg_er = torch.bincount(ei[0], minlength=200).float(); deg_inv_er = 1.0 / (deg_er + 1e-8)
    h_input = torch.randn(200, 16)
    model = ADRGNN(16, 16, 2, depth=64, dropout=0.0).cpu().eval()
    apsd_vals: List[float] = []
    with torch.no_grad():
        h0 = F.relu(model.in_lin(h_input))
        h = h0.clone()
        for layer in model.layers:
            h, _ = layer(h, h0, ei, deg_inv_er)
            apsd_vals.append(apsd(h))
            h = F.relu(h)
    for i in range(1, len(apsd_vals)):
        assert apsd_vals[i] >= 0.8 * apsd_vals[i - 1], f"APSD collapsed at layer {i}"

    # T4: gradient activity
    labels = torch.randint(0, 2, (200,))
    model.train(); opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        logits, _ = model(h_input, ei, deg_inv_er)
        loss = F.cross_entropy(logits, labels)
        loss.backward(); opt.step()
    g_eta = model.layers[0].eta.grad.abs().mean().item()
    g_gamma = model.layers[0].gamma.grad.abs().mean().item()
    assert g_eta > 1e-3 and g_gamma > 1e-3, "Gradients on η/γ too small – likely frozen"

    print("All Theory-Match assertions passed ✅")

    # Figure – eigenvalue histogram
    plt.figure(figsize=(4, 3))
    sns.histplot(eig.cpu().numpy(), bins=10, kde=False)
    plt.title("Eigenvalue spectrum (ring graph)")
    plt.xlabel("λ"); plt.ylabel("count")
    fname = os.path.join(FIG_DIR, "eigen_spectrum.pdf")
    plt.savefig(fname, bbox_inches="tight"); plt.close()
    print("Saved figure:", fname)

    res = {
        "T1_lin_diff": diff,
        "eig_min": eig.min().item(),
        "eig_max": eig.max().item(),
        "APSD_last": apsd_vals[-1],
        "grad_eta": g_eta,
        "grad_gamma": g_gamma,
    }
    print("\n==== EXP-1 NUMERICAL RESULTS ====)")
    print(json.dumps(res, indent=2))
    print("Figures: [eigen_spectrum.pdf]")

# -----------------------------------------------------------------------------
#  EXPERIMENT 2 – depth scaling benchmark (Cora demo)
# -----------------------------------------------------------------------------

def exp2() -> None:
    # (content unchanged)
    ...

# -----------------------------------------------------------------------------
#  EXPERIMENT 3 – placeholder (heavy benchmark)
# -----------------------------------------------------------------------------

def exp3() -> None:
    # (content unchanged)
    ...
