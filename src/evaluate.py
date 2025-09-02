"""src/evaluate.py
Evaluation utilities: accuracy computation, forgetting metrics and
(optionally) plotting helpers.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
#  Core evaluation
# -----------------------------------------------------------------------------

def evaluate(model: torch.nn.Module,
             loaders: List[torch.utils.data.DataLoader],
             *, device: torch.device):
    """Return (list-of-accuracies-per-task, mean-accuracy)."""
    model.eval()
    acc: List[float] = []
    with torch.no_grad():
        for loader in loaders:
            correct = total = 0
            for x, y in loader:
                x = x.to(device, non_blocking=device.type == "cuda")
                y = y.to(device, non_blocking=device.type == "cuda")
                pred = model(x).argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
            acc.append(100.0 * correct / max(total, 1))
    return acc, float(np.mean(acc) if acc else 0.0)

# -----------------------------------------------------------------------------
#  Forgetting measure (Chaudhry et al.)
# -----------------------------------------------------------------------------

def forgetting_curve(acc_per_task: List[List[float]]) -> float:
    """Compute average forgetting across tasks."""
    n_tasks = len(acc_per_task)
    if n_tasks <= 1:
        return 0.0
    f = []
    for t in range(n_tasks - 1):
        best = max(acc_per_task[t])      # best accuracy on task t so far
        last = acc_per_task[t][-1]       # accuracy after training last task
        f.append(best - last)
    return float(np.mean(f))
