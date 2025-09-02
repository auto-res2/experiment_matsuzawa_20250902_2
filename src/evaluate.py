"""src/evaluate.py
Evaluation/analysis helpers: memory measurement, confidence intervals, plotting.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

import matplotlib

# Headless backend so the code is CI-friendly
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set(style="whitegrid", font_scale=1.2)

# ----------------------------------------------------------------------------------
#  Low-level metrics helpers
# ----------------------------------------------------------------------------------

def peak_gpu_gb() -> float:
    """Return the peak CUDA memory in GB since the last reset (or NaN on CPU)."""
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated()
    return round(mem / 1024 ** 3, 3)


def ci95(arr: List[float]):
    """Mean and 95 % confidence interval of a list."""
    if len(arr) == 0:
        return float("nan"), float("nan")
    m = np.mean(arr)
    s = stats.sem(arr) if len(arr) > 1 else 0.0
    return m, 1.96 * s

# ----------------------------------------------------------------------------------
#  Figure helpers
# ----------------------------------------------------------------------------------

def save_bar(data: Dict[str, float], title: str, ylabel: str, fname: str):
    """Generate a simple labelled bar plot and store it as PDF."""
    if not data:
        warnings.warn("No data provided to save_bar(); figure will be skipped.")
        return

    sns.set_palette("Set2")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(data.keys(), data.values())
    for bar, val in zip(bars, data.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    Path(fname).with_suffix("")  # ensure parent path valid (race-free)
    plt.savefig(fname, bbox_inches="tight", format="pdf")
    plt.close()
