"""src/evaluate.py
Evaluation utilities: accuracy metrics, validation/test loop and plotting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train import accuracy, worst_group_accuracy  # reuse metric helpers

# Optional (only needed for figure plotting)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Validation / test loop
# ---------------------------------------------------------------------

def evaluate(model, loader: DataLoader) -> Tuple[float, float]:
    """Return (overall_acc, worst_group_acc).
    worst_group_acc = NaN if group labels absent.
    """
    model.eval()
    preds: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    gs: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            logits = model.forward_backbone(batch["x"].to(device))
            preds.append(logits.argmax(1).cpu())
            ys.append(batch["y"].cpu())
            if "g" in batch:
                gs.append(batch["g"].cpu())

    preds = torch.cat(preds)
    ys = torch.cat(ys)
    acc = accuracy(preds, ys)

    if len(gs) > 0:
        gs_cat = torch.cat(gs)
        wg_acc = worst_group_accuracy(preds, ys, gs_cat)
    else:
        wg_acc = float("nan")

    return acc, wg_acc

# ---------------------------------------------------------------------
# Simple plotting helper
# ---------------------------------------------------------------------

def plot_training_curves(history: Dict[str, List[float]], save_path: Path):
    epochs_arr = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=epochs_arr, y=history["train_loss"], label="Train loss")
    sns.lineplot(x=epochs_arr, y=history["val_acc"], label="Val acc")
    sns.lineplot(x=epochs_arr, y=history["val_wg"], label="Val worst-group")
    for x, y in zip(epochs_arr, history["val_acc"]):
        plt.text(x, y, f"{y:.1f}")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
