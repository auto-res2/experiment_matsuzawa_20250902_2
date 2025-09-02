"""src/evaluate.py
Evaluation utilities: metrics and a convenience wrapper to evaluate models.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from sklearn.metrics import accuracy_score

from .train import DTYPE  # reuse global setting

# -----------------------------------------------------------------------------
#  Metrics
# -----------------------------------------------------------------------------

def row_col_diff(x: torch.Tensor, y: torch.Tensor | None = None) -> Tuple[float, float]:
    """Compute Chen et al.'s row-diff & col-diff metrics.

    Args:
        x: node representations (N, F)
        y: optional labels (N,)
    Returns:
        row_diff, col_diff as floats.
    """
    if y is None:
        pairwise_l1 = torch.cdist(x, x, p=1)
        row_diff = pairwise_l1.mean().item()
    else:
        mask = (y.unsqueeze(0) == y.unsqueeze(1))
        d = torch.cdist(x.to(torch.float32), x.to(torch.float32), p=1)
        row_diff = d[mask].mean().item()
    col_diff = x.var(dim=0).mean().item()
    return row_diff, col_diff


# -----------------------------------------------------------------------------
#  Evaluation wrapper
# -----------------------------------------------------------------------------

def evaluate_model(model, data, masks: Dict[str, torch.Tensor]):
    """Run model in inference mode and return accuracy dictionary & smoothing metrics."""
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=-1)
    accs = {k: accuracy_score(data.y[m].cpu(), preds[m].cpu()) for k, m in masks.items()}
    rd, cd = row_col_diff(logits.float())
    return accs, rd, cd
