"""src/evaluate.py
Evaluation utilities: accuracy computation, validation loop, and plotting helpers.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # Headless backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from .train import accuracy  # Re-use shared metric

__all__ = ["evaluate", "save_line_fig", "save_bar_fig"]

# -----------------------------------------------------------------------------
#  Validation / test loop
# -----------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Return (avg_loss, avg_accuracy) on the given loader."""
    model.eval()
    acc_meter, loss_meter = [], []
    with torch.no_grad():
        for x, y, _, _ in tqdm(loader, desc="eval", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _, _, logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_meter.append(loss.item())
            acc_meter.append(accuracy(logits, y))
    return float(np.mean(loss_meter)), float(np.mean(acc_meter))


# -----------------------------------------------------------------------------
#  Plotting helpers
# -----------------------------------------------------------------------------

def save_line_fig(
    x: List[int],
    ys: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
):
    plt.figure(figsize=(6, 4))
    for name, y in ys.items():
        plt.plot(x, y, marker="o", label=name)
        for xi, yi in zip(x, y):
            plt.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    fname = f"{filename}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Figure saved: {fname}")
    plt.close()


def save_bar_fig(values: Dict[str, float], title: str, ylabel: str, filename: str):
    plt.figure(figsize=(8, 4))
    names, vals = list(values.keys()), list(values.values())
    bars = plt.bar(names, vals, color=sns.color_palette("husl", len(vals)))
    plt.title(title)
    plt.ylabel(ylabel)
    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = f"{filename}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Figure saved: {fname}")
    plt.close()
