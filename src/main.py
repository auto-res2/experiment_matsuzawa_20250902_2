"""src/main.py
Orchestrates Experiments 1–3 using refactored modules.
Run via:  python -m src.main
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision as tv
from torch.utils.data import DataLoader, Dataset

from .preprocess import (
    prepare_waterbirds,
    prepare_celeba,
    prepare_cifar_spurious,
)
from .train import (
    DualHead,
    SingleHead,
    Trainer,
    DEVICE,
    FIG_DIR,
    DATA_DIR,
    CKPT_DIR,
    SEEDS,
    set_seed,
)
from .evaluate import plot_bar

# --------------------------------------------------------------------
#  Experiment 1 – Standard benchmarks (ResNet-50)
# --------------------------------------------------------------------

def experiment1():
    print("\n============================================================")
    print("Experiment 1 – Standard Spurious-Correlation Benchmarks (RN-50)")
    print("============================================================")

    waterbirds = prepare_waterbirds()
    celeba = prepare_celeba()
    cifar = prepare_cifar_spurious()

    tasks: List[Tuple[str, Dict[str, Dataset], int]] = [
        ("Waterbirds", waterbirds, 2),
        ("CelebA-Hair", celeba, 2),
        ("CIFAR-Spurious", cifar, 10),
    ]

    results = {}
    for task_name, splits, num_cls in tasks:
        print(f"\n--- Task: {task_name} ---")
        train_loader = DataLoader(splits["train"], batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(splits["validation"], batch_size=64, shuffle=False, num_workers=4)
        test_loader = DataLoader(splits["test"], batch_size=64, shuffle=False, num_workers=4)

        set_seed(SEEDS[0])
        model = DualHead("resnet50", num_classes=num_cls)
        trainer = Trainer(model, train_loader, val_loader, num_epochs=50, experiment_name=f"ERM_{task_name}")
        trainer.run()
        acc_test = trainer.evaluate(test_loader)
        print(f"Test accuracy ({task_name}): {acc_test:.2f}%")
        results[task_name] = acc_test

    # ---------------- summary plot ---------------- #
    fig_path = FIG_DIR / "accuracy_erm.pdf"
    names = list(results.keys())
    accs = [results[n] for n in names]
    plot_bar(names, accs, "ERM baseline accuracy on three tasks", fig_path)


# --------------------------------------------------------------------
#  Experiment 2 – placeholder (dataset not public)
# --------------------------------------------------------------------

def experiment2():
    print("\n============================================================")
    print("Experiment 2 – Contextual Reliability Stress-Test (placeholder)")
    print("============================================================")
    print("Dataset Contextual-Waterbirds is not public yet – experiment skipped.")


# --------------------------------------------------------------------
#  Experiment 3 – Ablation & efficiency on CelebA 128×128
# --------------------------------------------------------------------

def experiment3():
    print("\n============================================================")
    print("Experiment 3 – Component Ablation & Efficiency (CelebA 128²)")
    print("============================================================")

    splits = prepare_celeba()

    # Down-scale to 128² on-the-fly
    class _Wrap(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, y, e = self.base[idx]
            x = tv.transforms.functional.resize(x, [128, 128])
            return x, y, e

    celeb_train = _Wrap(splits["train"])
    celeb_val = _Wrap(splits["validation"])
    celeb_test = _Wrap(splits["test"])

    loaders = {
        "train": DataLoader(celeb_train, batch_size=128, shuffle=True, num_workers=4),
        "val": DataLoader(celeb_val, batch_size=128, shuffle=False, num_workers=4),
        "test": DataLoader(celeb_test, batch_size=128, shuffle=False, num_workers=4),
    }

    # ---------------- baseline ERM ---------------- #
    model_erm = DualHead("resnet50", 2)
    trainer_erm = Trainer(model_erm, loaders["train"], loaders["val"], 30, "ERM_Celeb128", lr=5e-4)
    trainer_erm.run()
    acc_base = trainer_erm.evaluate(loaders["test"])
    print(f"Baseline ERM test-acc: {acc_base:.2f}%")

    # ---------------- ablation: single head ---------------- #
    model_single = SingleHead("resnet50", 2)
    trainer_single = Trainer(model_single, loaders["train"], loaders["val"], 30, "SingleHead_Celeb128", lr=5e-4)
    trainer_single.run()
    acc_single = trainer_single.evaluate(loaders["test"])
    print(f"Single-head ablation test-acc: {acc_single:.2f}%")

    # ---------------- plot ---------------- #
    fname = FIG_DIR / "ablation_dualhead.pdf"
    plot_bar(["ERM", "–DualHead"], [acc_base, acc_single], "Ablation – Dual-head importance (CelebA 128²)", fname)


# --------------------------------------------------------------------
#  Main entry point
# --------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA memory total: {mem_gb:.1f} GB")
    else:
        print("CUDA not available – running on CPU.")

    t0 = time.time()
    experiment1()
    experiment2()
    experiment3()
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
