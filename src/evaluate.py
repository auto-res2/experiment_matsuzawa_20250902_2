"""src/evaluate.py
Validation, statistics and simple plotting utilities.
"""
from __future__ import annotations
from typing import Tuple, List

import numpy as np
import torch
import seaborn as sns; sns.set(style="whitegrid", font_scale=1.1)  # noqa: E702
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.train import BF16_ENABLED, ci95  # reuse globals from train

# -----------------------------------------------------------------------------
# 1.  VALIDATION LOOP ----------------------------------------------------------
# -----------------------------------------------------------------------------

def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """Return (Top-1 %, Top-5 %) accuracy on *loader*."""

    model.eval()
    hits1 = hits5 = total = 0
    with torch.no_grad():
        for img, tgt in loader:
            img = img.cuda(non_blocking=True)
            tgt = tgt.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16_ENABLED):
                out = model(img)
            _, pred = out.topk(5, 1, True, True)
            total += tgt.size(0)
            hits1 += (pred[:, 0] == tgt).sum().item()
            hits5 += (pred == tgt.view(-1, 1)).sum().item()
    return hits1 / total * 100, hits5 / total * 100

# -----------------------------------------------------------------------------
# 2.  GENERIC PLOTTING HELPERS -------------------------------------------------
# -----------------------------------------------------------------------------

def bar_plot(fname: str, data: dict, title: str, ylab: str) -> None:
    """Create a 2-bar chart and write it to *fname* (PDF)."""

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(data.keys(), data.values(), color=["#1f77b4", "#ff7f0e"])
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel(ylab)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()
