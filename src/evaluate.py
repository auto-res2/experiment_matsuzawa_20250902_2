"""src/evaluate.py
Model evaluation utilities, statistical analysis, and experiment scripts.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Local imports (no top-level train import to avoid circular dependency)
from .preprocess import DATA_DIR, SplitCIFAR100
from .train import DEVICE, EFSBuffer, ResNet18_Stiefel, train_single_task

# -----------------------------------------------------------------------------
# Simple accuracy evaluation on a held-out set
# -----------------------------------------------------------------------------

def evaluate(backbone: torch.nn.Module, classifier: torch.nn.Module, ds) -> float:
    """Return top-1 accuracy (%) on *ds*."""
    backbone.eval()
    classifier.eval()
    loader = DataLoader(
        ds,
        batch_size=256,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = classifier(backbone(x))
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


# -----------------------------------------------------------------------------
# Experiment-1  :  Accuracy vs Memory Budget
# -----------------------------------------------------------------------------
RESULTS_COLS = ["dataset", "budget", "seed", "method", "AACC"]

BUDGETS = {"0.5MB": 500_000, "1MB": 1_000_000, "2MB": 2_000_000}


def run_exp1(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[tuple] = []

    for dataset_name in ["cifar100"]:
        for budget_name, B in BUDGETS.items():
            for seed in [11, 17]:  # Reduced seeds for demo purposes

                print("\n==============================")
                print(f"Experiment-1 | {dataset_name} | budget {budget_name} | seed {seed}")

                # ---------------- Dataset stream ----------------
                if dataset_name == "cifar100":
                    stream = SplitCIFAR100(DATA_DIR, seed)
                    n_tasks = 20
                else:
                    raise ValueError(f"Unknown dataset {dataset_name}")

                # Build shared backbone and buffer
                backbone = ResNet18_Stiefel()
                buffer = EFSBuffer(backbone.feature_dim, B_max=B)

                task_acc: List[float] = []
                for t in range(n_tasks):
                    tr_ds, val_ds, cls = stream.get_task_datasets(t)
                    classifier = torch.nn.Linear(backbone.feature_dim, len(cls)).to(DEVICE)
                    acc = train_single_task(
                        backbone,
                        classifier,
                        buffer,
                        tr_ds,
                        val_ds,
                        epochs=5,  # fewer epochs for a lightweight demo
                        batch_size=128,
                    )
                    task_acc.append(acc)
                    print(
                        f"Task {t:02d} acc {acc:.2f}% | buffer {len(buffer)} samples | {buffer.bytes_used} bytes"
                    )

                AACC = sum(task_acc) / len(task_acc)
                print(f"Final AACC {AACC:.2f}%")
                results.append((dataset_name, budget_name, seed, "EFS", AACC))

    # ---------------- Aggregate & save ----------------
    df = pd.DataFrame(results, columns=RESULTS_COLS)
    csv_path = out_dir / "exp1_results.csv"
    df.to_csv(csv_path, index=False)

    # ---------------- Plotting ----------------
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="budget", y="AACC", hue="dataset")
    for i, r in df.iterrows():
        plt.text(i, r.AACC + 0.5, f"{r.AACC:.1f}", ha="center")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Experiment-1: Accuracy vs Memory Budget")
    plt.legend()
    fig_path = out_dir / "accuracy_budget.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Saved figure → {fig_path.relative_to(out_dir.parent)}")
