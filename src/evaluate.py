"""
evaluate.py  –  statistical evaluation helpers & plotting utilities
"""
from __future__ import annotations

import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ptflops import get_model_complexity_info

# --------------------------------------------------------------------------------
#  FLOP counter
# --------------------------------------------------------------------------------

def compute_flops(model: torch.nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32)) -> float | None:
    """Return FLOPs (not MACs) for a single forward pass.
    If ptflops fails (e.g. unsupported layer), return None instead of crashing.
    """
    try:
        macs, _ = get_model_complexity_info(model, input_shape[1:], as_strings=False, print_per_layer_stat=False)
        return macs * 2.0  # 1 MAC = 2 FLOPs
    except Exception as e:  # pragma: no-cover –  safety first
        warnings.warn(f"ptflops could not analyse the model: {e}")
        return None

# --------------------------------------------------------------------------------
#  Small helper to annotate bar plots
# --------------------------------------------------------------------------------

def annotate_bar(ax):
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="bottom", fontsize=8)

# --------------------------------------------------------------------------------
#  Default plotting style (non-interactive backend set in main)
# --------------------------------------------------------------------------------

def set_plot_style():
    sns.set(style="whitegrid")
