"""
main.py – orchestrates MUCD toy experiment (Waterbirds) using refactored modules
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

from .train import ExperimentRunner, set_seed
from .preprocess import waterbirds_dataloaders
from .evaluate import save_bar_plot

# -----------------------------------------------------------------------------
# Device helper (used for FLOPs & latency measurement)
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    exp_description = (
        """\nExperiment-1 (toy run): Standard Robustness & Spurious-Gap Benchmarks on Waterbirds\n"
        "Model: ResNet-50 | Method: MUCD (2-epoch) | Mixed precision enabled if CUDA available\n"
        "Purpose: sanity-check end-to-end pipeline & figure generation. NOT full 90-epoch run.\n"""
    )
    print(exp_description)

    # reproducibility
    set_seed(2024)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = waterbirds_dataloaders(batch_size=32)

    # ------------------------------------------------------------------
    # Training (short 2-epoch demo)
    # ------------------------------------------------------------------
    runner = ExperimentRunner("waterbirds_mucd", num_classes=2, epochs=2)
    runner.train(train_loader, val_loader)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    test_acc = runner.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

    # ------------------------------------------------------------------
    # FLOPs & latency measurement (single mini-batch)
    # ------------------------------------------------------------------
    sample = next(iter(test_loader))[0].to(DEVICE)
    flops = FlopCountAnalysis(runner.model, sample).total()
    flops_g = flops / 1e9

    if DEVICE.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            starter.record()
            _ = runner.model(sample)
            ender.record()
            torch.cuda.synchronize()
            latency_ms = starter.elapsed_time(ender) / len(sample)  # ms / img
    else:
        import time

        start = time.time()
        with torch.no_grad():
            _ = runner.model(sample)
        latency_ms = (time.time() - start) * 1000 / len(sample)

    metrics = {"Accuracy": test_acc, "Latency(ms)": latency_ms, "GFLOPs": flops_g}
    print(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    save_bar_plot({"MUCD": test_acc}, "Test Accuracy – Waterbirds (toy)", "accuracy_waterbirds.pdf")


if __name__ == "__main__":
    main()
