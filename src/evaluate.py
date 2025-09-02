"""
evaluate.py
Validation utilities: evaluation loop, worst-group accuracy and figure helpers.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # Head-less backend for clusters
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402; kept for compatibility even if not used directly

# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item() * 100.0


def worst_group_acc(meta: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> float:
    """Waterbirds: group = 2*y + place"""
    place = meta[:, 1]
    groups = 2 * y + place
    preds = logits.argmax(1)
    worst = 100.0
    for g in range(4):
        mask = groups == g
        if mask.sum() == 0:
            continue
        group_acc = (preds[mask] == y[mask]).float().mean().item() * 100.0
        worst = min(worst, group_acc)
    return worst


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses, accs, wg_accs = [], [], []
    with torch.no_grad():
        for x, y, meta, _ in tqdm(loader, desc="eval", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _feat, _proj, logits = model(x)
            loss = F.cross_entropy(logits, y)
            losses.append(loss.item())
            accs.append(accuracy_from_logits(logits, y))
            wg_accs.append(worst_group_acc(meta, logits, y))

    return {
        "val_loss": float(np.mean(losses)),
        "val_acc": float(np.mean(accs)),
        "val_worst_group_acc": float(np.mean(wg_accs)),
    }


# -----------------------------------------------------------------------------
# Figure utilities
# -----------------------------------------------------------------------------

def save_line_plot(
    x: List[int],
    ys: Dict[str, List[float]],
    title: str,
    ylab: str,
    fname: str,
) -> None:
    plt.figure(figsize=(6, 4))
    for k, v in ys.items():
        plt.plot(x, v, marker="o", label=k)
        for xi, yi in zip(x, v):
            plt.annotate(
                f"{yi:.1f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
            )
    plt.xlabel("Epoch")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pdf = f"{fname}.pdf"
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] saved {pdf}")
