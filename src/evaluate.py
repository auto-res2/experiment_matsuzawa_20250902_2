"""src/evaluate.py
Evaluation & plotting utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # non-GUI back-ends (e.g. servers)
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["plot_bar"]


def plot_bar(names: List[str], accs: List[float], title: str, save_path: Path):
    """Generic bar-plot helper used by Experiment 1 & 3."""
    plt.figure(figsize=(6, 4))
    sns.barplot(x=names, y=accs, palette="deep")
    for i, a in enumerate(accs):
        plt.text(i, a + 0.5, f"{a:.1f}", ha="center")
    plt.ylabel("Test Accuracy %")
    plt.title(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print("Figure saved:", save_path)
