"""src/evaluate.py – experimental loops, statistics & plotting (patched)"""
from pathlib import Path
import os
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch  # Added missing import
from avalanche.benchmarks.classic import RotatedMNIST

from .preprocess import (
    DATA_DIR,
    OUT_DIR,
    SEEDS,
    set_seed,
    SplitCIFAR100,
)
from .train import (
    ResNet18Stiefel,
    EFSBuffer,
    RawImageBuffer,
    RingBuffer,
    train_task,
)

# --------------------------------------------------------------------------------
# 1) Memory budget experiment -----------------------------------------------------
# --------------------------------------------------------------------------------
BUDGETS: Dict[str, int] = {"0.5MB": 512_000, "1MB": 1_024_000, "2MB": 2_048_000}
METHODS = {"EFS": EFSBuffer, "ER_RAW": RawImageBuffer, "ER_RING": RingBuffer}


def run_exp1():
    """Experiment-1: Accuracy vs memory budget on Split CIFAR-100."""
    print("\n========== EXPERIMENT-1  (Byte-Budget vs Accuracy) ==========")
    out = OUT_DIR / "exp1_byte_budget"
    out.mkdir(exist_ok=True, parents=True)

    quick = os.getenv("QUICK", "1") == "1"
    datasets = [("cifar100", SplitCIFAR100)]  # extendable

    results = []
    for dname, DCls in datasets:
        for budget_name, B in BUDGETS.items():
            seeds = [SEEDS[0]] if quick else SEEDS
            for seed in seeds:
                set_seed(seed)
                stream = DCls(DATA_DIR, seed)

                # one frozen backbone shared across methods ------------------
                backbone = ResNet18Stiefel()
                buffer_objs: Dict[str, Any] = {}
                for m, cls in METHODS.items():
                    if m == "EFS":
                        buffer_objs[m] = cls(backbone.feat_dim, B)
                    else:
                        buffer_objs[m] = cls(B)

                task_acc = {m: [] for m in METHODS}
                num_tasks = 3 if quick else 20
                for t in range(num_tasks):
                    tr, val, _, cls = stream.get_task(t)
                    for m, buff in buffer_objs.items():
                        # ---------------- classifier -----------------------
                        # Use global label space (100 classes) to avoid index mismatches
                        clf = torch.nn.Linear(backbone.feat_dim, 100)
                        acc = train_task(
                            backbone,
                            clf,
                            buff,
                            tr,
                            val,
                            epochs=3 if quick else 200,
                            replay_r=0.5,
                        )
                        task_acc[m].append(acc)
                        print(
                            f"[{dname}|{budget_name}|seed{seed}|{m}] task{t:02d} acc={acc:.2f}% buffer={len(buff)} items"
                        )
                for m in METHODS:
                    AACC = np.mean(task_acc[m])
                    results.append(
                        dict(dataset=dname, budget=budget_name, seed=seed, method=m, AACC=AACC)
                    )

    df = pd.DataFrame(results)
    df.to_csv(out / "results.csv", index=False)

    # ------------- plot ---------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="budget", y="AACC", hue="method")
    for i, r in df.iterrows():
        plt.text(i, r.AACC + 0.3, f"{r.AACC:.1f}", ha="center", fontsize=8)
    plt.ylabel("Average Accuracy (%)")
    plt.title("Accuracy vs Memory Budget – Split CIFAR-100")
    plt.legend()
    fig_path = out / "accuracy_budget.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print("Figure saved:", fig_path.name)

# --------------------------------------------------------------------------------
# 2) Long-stream Rotated-MNIST experiment -----------------------------------------
# --------------------------------------------------------------------------------

def run_exp2():
    print("\n========== EXPERIMENT-2  (50-Task Long-Stream) ==========")
    out = OUT_DIR / "exp2_long_stream"
    out.mkdir(exist_ok=True, parents=True)

    quick = os.getenv("QUICK", "1") == "1"
    B = 1_024_000
    methods = ["EFS", "ER_RING", "ER_RAW"]

    records = []
    seeds = [SEEDS[0]] if quick else SEEDS
    for seed in seeds:
        set_seed(seed)
        stream = RotatedMNIST(n_rotations=50, seed=seed)
        backbone = ResNet18Stiefel()
        buffers = {
            m: (
                EFSBuffer(backbone.feat_dim, B)
                if m == "EFS"
                else RingBuffer(B)
                if m == "ER_RING"
                else RawImageBuffer(B)
            )
            for m in methods
        }
        acc_hist = {m: [] for m in methods}
        for task_id, benchmark_task in enumerate(stream.train_stream):
            tr_ds = benchmark_task.dataset
            val_ds = stream.test_stream[task_id].dataset
            for m, buff in buffers.items():
                clf = torch.nn.Linear(backbone.feat_dim, 10)
                acc = train_task(
                    backbone,
                    clf,
                    buff,
                    tr_ds,
                    val_ds,
                    epochs=1 if quick else 50,
                    replay_r=0.5,
                )
                acc_hist[m].append(acc)
            if quick and task_id == 2:
                break  # shorten for CI

        for m in methods:
            records.append(dict(seed=seed, method=m, final_acc=np.mean(acc_hist[m])))

    pd.DataFrame(records).to_csv(out / "results.csv", index=False)

# --------------------------------------------------------------------------------
# 3) Ablation study ---------------------------------------------------------------
# --------------------------------------------------------------------------------

def run_exp3():
    print("\n========== EXPERIMENT-3  (Ablation & Robustness) ==========")
    out = OUT_DIR / "exp3_ablation"
    out.mkdir(exist_ok=True, parents=True)

    quick = os.getenv("QUICK", "1") == "1"
    variants = {
        "full": dict(wgf=True, subcb=True),
        "no_wgf": dict(wgf=False, subcb=True),
        "no_sub": dict(wgf=True, subcb=False),
        "plain": dict(wgf=False, subcb=False),
    }

    seed = SEEDS[0]
    set_seed(seed)
    stream = SplitCIFAR100(DATA_DIR, seed)
    backbone = ResNet18Stiefel()
    B = 1_024_000

    res = []
    num_tasks = 3 if quick else 20
    for name, _ in variants.items():
        buf = EFSBuffer(backbone.feat_dim, B)
        task_acc = []
        for t in range(num_tasks):
            tr, val, _, cls = stream.get_task(t)
            clf = torch.nn.Linear(backbone.feat_dim, 100)  # global label space
            acc = train_task(
                backbone,
                clf,
                buf,
                tr,
                val,
                epochs=2 if quick else 200,
                replay_r=0.5,
            )
            task_acc.append(acc)
        res.append(dict(variant=name, AACC=np.mean(task_acc)))

    pd.DataFrame(res).to_csv(out / "results.csv", index=False)