"""src/evaluate.py
Evaluation / analysis utilities extracted from the original experimental script.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
import matplotlib

matplotlib.use("Agg")  # head-less backends for servers / CI
import matplotlib.pyplot as plt

sns.set(style="whitegrid", font_scale=1.3)

# ---------------------------------------------------------------------------- #
# Core evaluation helpers
# ---------------------------------------------------------------------------- #

def memory_benchmark(model: nn.Module, batch_size: int = 8, device: str = "cuda") -> float:
    """Return peak CUDA memory in **GB** for a single forward pass."""
    if device == "cpu" or not torch.cuda.is_available():
        # Cannot measure accurately on CPU; return NaN so downstream stats ignore it
        return float("nan")

    model.eval()
    torch.cuda.reset_peak_memory_stats()

    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        model(dummy)

    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1024**3
    return round(mem, 3)


def evaluate(model: nn.Module, loader) -> float:
    """Top-1 accuracy (%) on supplied loader."""
    device = next(model.parameters()).device
    model.eval()
    hits = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(imgs).argmax(1)
            hits += (pred == labels).sum().item()
            total += labels.numel()
    return 100.0 * hits / total

# ---------------------------------------------------------------------------- #
# Statistical helpers & visualisation
# ---------------------------------------------------------------------------- #

def ci95(arr: List[float]) -> Tuple[float, float]:
    m = float(np.mean(arr))
    sem = float(stats.sem(arr))
    ci = 1.96 * sem if len(arr) > 1 else 0.0
    return m, ci


def ttest(a: List[float], b: List[float]) -> float:
    return float(stats.ttest_rel(a, b).pvalue) if len(a) == len(b) else float("nan")


def print_bar(title: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")


def save_bar_plot(data: Dict[str, float], ylabel: str, title: str, fname: str) -> None:
    keys, vals = list(data.keys()), list(data.values())
    palette = sns.color_palette("Set2", len(keys))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals, color=palette)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.legend(keys, frameon=False)
    plt.savefig(fname, bbox_inches="tight", format="pdf")
    plt.close()

# ---------------------------------------------------------------------------- #
# Convenience for Exp-2 (model memory fitting)
# ---------------------------------------------------------------------------- #

def try_init_model(depth: int, dim: int, block_cls, device: str):
    """Attempt to instantiate & benchmark a model.  Returns (fits, peak_ram)."""
    from train import TinyVisionMamba  # local import to avoid circularity

    try:
        model = TinyVisionMamba(block_cls, depth=depth, dim=dim).to(device)
        mem = memory_benchmark(model, batch_size=8, device=device)
        return True, mem
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return False, None
        raise exc
