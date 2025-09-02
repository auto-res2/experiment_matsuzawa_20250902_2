"""src/evaluate.py
---------------------------------------------------------------------
Contains experiment definitions, statistical analysis, and plotting
routines.  Each experiment orchestrates dataset preparation, buffer
instantiation, training, and finally the creation of publication-quality
figures.
"""

from __future__ import annotations

# --------------------------- std-lib -------------------------------
import textwrap
from typing import List, Dict, Any

# -------------------------- third-party ----------------------------
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # headless/non-interactive backend
import matplotlib.pyplot as plt
import torch
from torchvision import models

# Avalanche benchmark generators
from avalanche.benchmarks.classic import SplitCIFAR100

# -------------------------- local imports -------------------------
from .preprocess import (
    DATA_ROOT,
    FIG_ROOT,
    LOG_ROOT,
    download_and_extract,
)
from .train import (
    BitPackBuffer,
    RawRingBuffer,
    train_experience_replay,
)

# ------------------------------------------------------------------
#  Helper – robust metric extraction (Avalanche API changed over time)
# ------------------------------------------------------------------

def _find_metric(metrics: Dict[str, Any], name_substr: str) -> float:
    """Return the first metric whose key contains *name_substr* (case-insensitive).

    The Avalanche metric naming scheme changed between versions.  To
    remain compatible across releases we therefore search for the key
    at runtime instead of hard-coding the full string.  Preference is
    given to metrics collected during the *eval* phase when multiple
    matches exist.
    """

    # Sort keys to ensure deterministic retrieval order
    keys = sorted(metrics.keys())
    preferred, fallback = None, None
    for k in keys:
        if name_substr.lower() in k.lower():
            if "/eval" in k or "eval_phase" in k:
                preferred = k
            else:
                fallback = k if fallback is None else fallback
    key = preferred or fallback
    if key is None:
        raise KeyError(f"Metric containing '{name_substr}' not found. Available keys: {list(metrics)}")
    return float(metrics[key])


# ==================================================================
#  Experiment 1 – Memory × Accuracy trade-off on Split CIFAR-100
# ==================================================================

def run_experiment_1():
    print(
        textwrap.dedent(
            """
            ------------------------------------------------------------
            EXPERIMENT 1 – Memory × Accuracy Trade-off Benchmarking
            ------------------------------------------------------------
            Goal: Compare Bit-Pack Replay (BPR) against a raw-image ring
            buffer under aggressive memory budgets {2, 5, 10, 50} kB per
            task on Split CIFAR-100.
            """
        )
    )

    # ---------------- dataset download / benchmark ----------------
    download_and_extract(
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", DATA_ROOT
    )
    benchmark = SplitCIFAR100(
        n_experiences=10,
        seed=0,
        fixed_class_order=None,
        return_task_id=True,
    )

    # ---------------- experiment hyper-params ----------------------
    img_shape = (3, 32, 32)
    budgets_kb = [2, 5, 10, 50]
    methods: List[str] = ["BPR", "RAW"]
    all_records = []

    for budget in budgets_kb:
        for method in methods:
            torch.cuda.empty_cache()
            print(f"\n*** Budget = {budget} kB | Method = {method} ***")

            if method == "BPR":
                buffer = BitPackBuffer(img_shape, mem_budget_kb=budget, rank=32)
            elif method == "RAW":
                buffer = RawRingBuffer(img_shape, mem_budget_kb=budget)
            else:
                raise ValueError(method)

            # CIFAR-friendly ResNet-18 (kernel 3 stride 1, no max-pool)
            model = models.resnet18(num_classes=100)
            model.conv1 = torch.nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.maxpool = torch.nn.Identity()

            res = train_experience_replay(
                model,
                benchmark,
                buffer,
                n_epochs_per_task=1,  # keep runtime reasonable for demo
                batch_size=64,
                lr=0.1,
            )

            # ------------------------------------------------------
            # Robustly extract final average accuracy & forgetting
            # ------------------------------------------------------
            final_metrics = res[-1]
            final_acc = _find_metric(final_metrics, "Top1_Accuracy_Stream") * 100.0
            final_forget = _find_metric(final_metrics, "StreamForgetfulness")

            print(
                f"Budget {budget} kB | {method} → Final Acc = {final_acc:.2f} % | "
                f"Avg Forget = {final_forget:.4f}"
            )
            all_records.append(
                {
                    "budget_kb": budget,
                    "method": method,
                    "final_acc": final_acc,
                    "forgetting": final_forget,
                }
            )

    # ---------------- save CSV & generate figures ------------------
    df = pd.DataFrame(all_records)
    csv_path = LOG_ROOT / "exp1_results.csv"
    df.to_csv(csv_path, index=False)

    # ----- Accuracy vs Memory budget -----
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="budget_kb", y="final_acc", hue="method", marker="o")
    plt.title("Accuracy vs Memory budget (Split CIFAR-100)")
    plt.xlabel("Memory per task [kB]")
    plt.ylabel("Final average accuracy [%]")
    for _, row in df.iterrows():
        plt.text(row.budget_kb, row.final_acc + 0.5, f"{row.final_acc:.1f}", ha="center")
    plt.xscale("log")
    plt.xticks(budgets_kb, budgets_kb)
    plt.legend()
    acc_fig = FIG_ROOT / "accuracy_memtradeoff.pdf"
    plt.savefig(acc_fig, bbox_inches="tight")
    print(f"Saved figure: {acc_fig}")

    # ----- Forgetting vs Memory budget -----
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="budget_kb", y="forgetting", hue="method", marker="o")
    plt.title("Average Forgetting vs Memory budget (Split CIFAR-100)")
    plt.xlabel("Memory per task [kB]")
    plt.ylabel("Average forgetting")
    for _, row in df.iterrows():
        plt.text(row.budget_kb, row.forgetting + 0.001, f"{row.forgetting:.3f}", ha="center")
    plt.xscale("log")
    plt.xticks(budgets_kb, budgets_kb)
    plt.legend()
    forget_fig = FIG_ROOT / "forgetting_memtradeoff.pdf"
    plt.savefig(forget_fig, bbox_inches="tight")
    print(f"Saved figure: {forget_fig}")

    # ---------------- console summary ------------------------------
    print("\nNumerical results (single seed):")
    print(df.to_string(index=False))
    print("Figures generated:\n  – accuracy_memtradeoff.pdf\n  – forgetting_memtradeoff.pdf")
