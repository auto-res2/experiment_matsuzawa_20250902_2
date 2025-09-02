"""src/evaluate.py – evaluation utilities, statistics & plotting."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # head-less backend for servers/CI
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 – imported for styling side-effects
import torch
import torch.nn.functional as F

from .train import DEVICE, DTYPE  # shared constants

# -----------------------------------------------------------------------------
# Project directories – only FIG_DIR is needed here ---------------------------
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Embedding statistics ---------------------------------------------------------
# -----------------------------------------------------------------------------

def layer_norm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:])


def row_col_diff(emb: torch.Tensor) -> Tuple[float, float]:
    """Row / column difference metrics to monitor over-smoothing."""

    x = layer_norm(emb.float())
    row_diff = torch.cdist(x, x, p=1).mean().item()
    col_diff = x.var(dim=0, unbiased=False).mean().item()
    # clip for readability in plots
    return min(row_diff, 1e3), min(col_diff, 1e3)


# -----------------------------------------------------------------------------
# Main public helpers ----------------------------------------------------------
# -----------------------------------------------------------------------------

def evaluate(model, data, masks):  # type: ignore
    """Run model in eval mode and compute accuracy + over-smoothing metrics."""

    from sklearn.metrics import accuracy_score  # local import => avoids heavy dep at import time
    model.eval()

    autocast_ctx = (
        torch.cuda.amp.autocast(device_type="cuda", dtype=DTYPE)
        if DEVICE.type == "cuda"
        else torch.cpu.amp.autocast(enabled=False)  # type: ignore – dummy ctx
    )
    with torch.no_grad(), autocast_ctx:
        logits = model(data.x, data.edge_index).float()

    pred = logits.argmax(dim=-1)
    accs = {
        k: accuracy_score(data.y[m].cpu(), pred[m].cpu()) for k, m in masks.items()
    }
    rd, cd = row_col_diff(logits)
    return accs, rd, cd


# ----------------------------------

def save_accuracy_plot(
    depths: Sequence[int],
    accs: Dict[str, List[float]],
    ds_name: str,
):
    """Render a PDF that shows depth vs. accuracy for the given dataset."""

    plt.figure(figsize=(6, 4))
    for model_name, ys in accs.items():
        plt.plot(depths, ys, marker="o", label=model_name)
        for x, y in zip(depths, ys):
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.xscale("log", basex=2)
    plt.xlabel("#Layers")
    plt.ylabel("Accuracy")
    plt.title(f"Depth-Accuracy – {ds_name}")
    plt.legend()
    fname = FIG_DIR / f"accuracy_{ds_name}.pdf"
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[FIG] Saved {fname.relative_to(ROOT)}")


__all__ = [
    "evaluate",
    "save_accuracy_plot",
]
