"""src/main.py
Orchestrates the end-to-end experimental workflow.  Run with
    python -m src.main
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Local imports ---------------------------------------------------------------
from src.train import (set_seed, ResNet18Feat, train_single_task,
                       ERBuffer, HOFQBuffer)
from src.evaluate import evaluate, forgetting_curve
from src.preprocess import get_split_cifar100

# -----------------------------------------------------------------------------
#  House-keeping: output directories
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures";  FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
#  Single experiment (Experiment-1 from the paper)
# -----------------------------------------------------------------------------

def experiment1(*, split_root: str = "./data", seeds: List[int] | None = None,
                device: torch.device):
    if seeds is None:
        seeds = [0]
    print("\n=== EXPERIMENT-1 – 'How far can 5 MB go?' ===\n")
    methods: Dict[str, Dict] = {
        "hofq"    : dict(buffer_kb=200,  cls=HOFQBuffer),
        "er_ring" : dict(buffer_imgs=130, cls=ERBuffer),
        "er_full" : dict(buffer_imgs=2000,cls=ERBuffer),
        "finetune": dict(buffer_imgs=0,   cls=None),
    }

    results: Dict[str, List[dict]] = {m: [] for m in methods}

    for seed in seeds:
        set_seed(seed)
        train_tasks, test_tasks = get_split_cifar100(root=split_root)

        for m_key, cfg in methods.items():
            model = ResNet18Feat(num_classes=100).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_tasks))

            # ------------------------------------------------------------------
            if m_key == "hofq":
                buffer = HOFQBuffer(capacity_kb=cfg["buffer_kb"], bytes_per_sample=2, device=device)
            elif m_key.startswith("er"):
                buffer = ERBuffer(capacity=cfg["buffer_imgs"], device=device)
            else:
                buffer = None

            # ------------------------------------------------------------------
            acc_matrix: List[List[float]] = []
            for t, task_ds in enumerate(train_tasks):
                loader = DataLoader(task_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=device.type=="cuda")
                if m_key == "hofq":
                    buffer.new_task()
                train_single_task(model, loader, optimizer, criterion, m_key, buffer, t, device=device)
                scheduler.step()

                # evaluate on *seen* tasks so far
                seen_loaders = [DataLoader(d, batch_size=64, shuffle=False, num_workers=2, pin_memory=device.type=="cuda")
                                 for d in test_tasks[:t+1]]
                _, mean_acc = evaluate(model, seen_loaders, device=device)
                acc_matrix.append([mean_acc])

            # final evaluation on all tasks
            all_loaders = [DataLoader(d, batch_size=64, shuffle=False, num_workers=2, pin_memory=device.type=="cuda")
                           for d in test_tasks]
            per_task_acc, last_acc = evaluate(model, all_loaders, device=device)
            avg_forget = forgetting_curve(acc_matrix)
            results[m_key].append(dict(acc_last=last_acc, forget=avg_forget))

            # tidy-up CUDA memory
            del model; torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Aggregate & show results
    # -------------------------------------------------------------------------
    table = []
    for m, runs in results.items():
        accs = [r["acc_last"] for r in runs]
        fgs  = [r["forget"]   for r in runs]
        table.append([m, np.mean(accs), np.std(accs), np.mean(fgs)])
    df = pd.DataFrame(table, columns=["method", "ACC_last", "std", "Forget"])
    print(df.to_string(index=False, float_format="%.2f"))

    # bar-plot
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="method", y="ACC_last", data=df, ax=ax, palette="tab10")
    ax.set_ylim(0, 100)
    ax.set_ylabel("ACC_last (%)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width()/2., p.get_height()+1),
                    ha='center', va='bottom')
    ax.set_title("Exp-1 : Final accuracy under 5 MB cap")
    fig.savefig(FIG_DIR/"accuracy_baselines.pdf", bbox_inches="tight")
    print("Saved figure   →", FIG_DIR/"accuracy_baselines.pdf")

# -----------------------------------------------------------------------------
#  Main entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running continual-learning experiments on {device}!\n")

    # A quick demo only runs Experiment-1 for speed.  Uncomment the others to
    # reproduce the full paper (will take hours).
    experiment1(device=device)

    print("Done – figures are in ./figures and tables printed above.")
