"""
evaluate.py – evaluation utilities (plots, metrics)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import matplotlib

matplotlib.use("Agg")  # headless back-end suitable for servers
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Simple bar plot helper used in toy experiment
# -----------------------------------------------------------------------------

SAVE_DIR = Path(".research/iteration2/images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_bar_plot(data: Dict[str, float], title: str, filename: str):
    """Save a bar plot (PDF) for provided metric dictionary.

    Args:
        data: mapping from label -> scalar value (e.g., accuracy).
        title: plot title.
        filename: output PDF path (basename will be placed inside SAVE_DIR).
    """
    out_file = SAVE_DIR / filename
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(data.keys()), y=list(data.values()), palette="viridis")
    for i, v in enumerate(data.values()):
        plt.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated figure: {out_file.relative_to(Path.cwd())}")
