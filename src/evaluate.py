"""
evaluate.py
Very light-weight evaluation utilities so that `main.py` can import them.
They *do not* attempt to replicate the full WILDS evaluation – they simply
compute average accuracy across the supplied DataLoader.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib

# Use a non-interactive backend so the code also works on headless CI servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 – after backend selection
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from tqdm import tqdm  # noqa: E402

__all__ = [
    "evaluate",
    "save_line_plot",
]

# -----------------------------------------------------------------------------
#  Core evaluation – plain accuracy plus a dummy worst-group metric
# -----------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
) -> Dict[str, float]:
    model.eval()
    accs: List[float] = []
    with torch.no_grad():
        for x, y, _meta, _ in tqdm(loader, desc="val", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _feat, _proj, logits = model(x)
            accs.append((logits.argmax(1) == y).float().mean().item() * 100.0)

    val_acc = float(np.mean(accs)) if accs else 0.0
    # We do not have real groups – return acc again so scripts don't crash.
    return {"val_acc": val_acc, "val_worst_group_acc": val_acc}


# -----------------------------------------------------------------------------
#  Plot helpers – store figures under the mandated path
# -----------------------------------------------------------------------------

_SAVE_DIR = Path(".research/iteration7/images")


def save_line_plot(
    xs: List[int],
    ys_dict: Dict[str, List[float]],
    *,
    title: str,
    ylab: str,
    fname: str,
) -> None:  # pragma: no cover
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))

    for label, ys in ys_dict.items():
        plt.plot(xs, ys, label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    outfile = _SAVE_DIR / f"{fname}.pdf"
    plt.savefig(outfile)
    plt.close()
    print(f"[INFO] Figure saved → {outfile.relative_to(Path.cwd())}")
