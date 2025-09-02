"""src/evaluate.py
Evaluation helpers: memory benchmark, accuracy calculation and plotting
utilities.
"""
from __future__ import annotations

import typing as _t

import numpy as np  # noqa: F401 – reserved for future use
import torch
import matplotlib

# Use a head-less backend suitable for server environments BEFORE importing
# pyplot.  This avoids the need for an X-server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import seaborn as sns  # noqa: E402  pylint: disable=wrong-import-position

sns.set(style="whitegrid", font_scale=1.3)

__all__ = [
    "memory_benchmark",
    "evaluate",
    "bar_plot",
]


# -----------------------------------------------------------------------------
# 1.  GPU memory benchmark
# -----------------------------------------------------------------------------

@torch.no_grad()
def memory_benchmark(
    model: torch.nn.Module,
    img_size: int = 224,
    start_bs: int = 16,
    max_bs: int = 128,
) -> int:
    """Find the largest batch-size that fits into GPU memory without OOM.

    Because we only use placeholder models, the function also caps absolute
    batch size to avoid needlessly allocating gigabytes of memory.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    bs = start_bs
    last_good = bs
    while bs <= max_bs:
        try:
            dummy = torch.randn(bs, 3, img_size, img_size, device=device)
            _ = model(dummy)  # forward pass – we only care about memory usage
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            last_good = bs
            bs *= 2
        except RuntimeError as err:  # pragma: no cover – hardware dependent
            if "out of memory" in str(err).lower():
                break
            raise err
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return last_good


# -----------------------------------------------------------------------------
# 2.  Accuracy evaluation
# -----------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader: "torch.utils.data.DataLoader") -> float:  # noqa: D401, F722
    """Top-1 accuracy in percent."""

    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    return 100.0 * correct / total if total else 0.0


# -----------------------------------------------------------------------------
# 3.  Plotting helpers
# -----------------------------------------------------------------------------

def bar_plot(
    data: dict[str, float],
    ylabel: str,
    title: str,
    fname: str,
) -> None:
    """Horizontal bar plot with values annotated on top of the bars."""

    keys = list(data.keys())
    values = list(data.values())
    palette = sns.color_palette("Set2", len(keys))

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, values, color=palette)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    try:
        plt.savefig(fname, bbox_inches="tight", format="pdf")
    except Exception as err:  # pragma: no cover – I/O
        print(f"[Warning] Could not save figure {fname}: {err}")
    finally:
        plt.close(fig)
