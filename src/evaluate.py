"""
evaluate.py – thin wrapper around the MetricLogger evaluation utilities.
Only functions already present in the original monolithic file are exposed
here so that they can be imported cleanly from other modules.
"""
from __future__ import annotations
from typing import Dict, Iterable
import torch

from src.metrics import MetricLogger

__all__ = ["evaluate"]

def evaluate(model, dataloader, device: torch.device, meter: MetricLogger) -> Dict[str, Iterable]:
    """Run evaluation split and return the metric dictionary produced by
    `meter.eval_split` from the original implementation."""
    model.eval()
    with torch.no_grad():
        return meter.eval_split(model, dataloader, device)
