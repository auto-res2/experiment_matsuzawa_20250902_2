"""src/evaluate.py
Evaluation and plotting helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.cuda.amp import autocast

__all__ = [
    "evaluate_cls",
    "save_lineplot",
    "save_barplot",
]

# ---------------------------------------------------------------------------
#  RENDERING BACK-END & OUTPUT LOCATION
# ---------------------------------------------------------------------------

matplotlib.use("Agg")  # ensure head-less back-end

# All figures must live under this directory (created once on import).
_OUTPUT_DIR = Path(".research/iteration2/images")
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  CLASSIFICATION EVALUATION
# ---------------------------------------------------------------------------

def evaluate_cls(model: torch.nn.Module, loader) -> float:
    """Return Top-1 accuracy (%) over *loader* for *model*."""

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True).float()
            y = y.cuda(non_blocking=True)
            with autocast(dtype=torch.float16):
                logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
#  PLOTTING HELPERS
# ---------------------------------------------------------------------------

def _to_outpath(fname: str | Path) -> Path:
    """Return the absolute path under the designated output directory."""
    p = Path(fname)
    return _OUTPUT_DIR / p.name  # ensure flat namespace inside the folder


def save_lineplot(
    x: List,
    ys: Dict[str, List],
    title: str,
    xlabel: str,
    ylabel: str,
    fname: str | Path,
):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    for label, y in ys.items():
        plt.plot(x, y, marker="o", label=label)
        for xi, yi in zip(x, y):
            plt.text(xi, yi, f"{yi:.1f}", fontsize=8, ha="center", va="bottom")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    out = _to_outpath(fname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def save_barplot(
    categories: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    fname: str | Path,
):
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=categories, y=values, palette="pastel")
    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out = _to_outpath(fname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
