"""
preprocess.py  –  dataset / benchmark builders and reproducibility helpers
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet
from avalanche.benchmarks.datasets import OmniglotDataset
from avalanche.benchmarks import NCScenario

__all__ = [
    "set_all_seeds",
    "build_cifar100_benchmark",
    "build_tinyimagenet_benchmark",
    "build_omniglot_rotation_scenario",
]

# =============================================================================
#  Reproducibility helper
# =============================================================================

def set_all_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
#  Benchmark / Scenario builders
# =============================================================================

def build_cifar100_benchmark(n_experiences: int = 20, *, return_task_id: bool = False):
    return SplitCIFAR100(n_experiences, return_task_id=return_task_id)


def build_tinyimagenet_benchmark(n_experiences: int = 10, *, return_task_id: bool = False):
    return SplitTinyImageNet(n_experiences, return_task_id=return_task_id)


def build_omniglot_rotation_scenario(num_tasks: int = 100):
    """Create a Permuted-/Rotated-Omniglot scenario, one rotation per task."""
    omniglot_train = OmniglotDataset(train=True, download=True)
    omniglot_test = OmniglotDataset(train=False, download=True)

    tasks_train, tasks_test = [], []
    for t in range(num_tasks):
        rot_angle = t * (360.0 / num_tasks)
        rot = T.RandomRotation((rot_angle, rot_angle))
        tfm = T.Compose([rot, T.Resize((28, 28)), T.ToTensor()])
        tasks_train.append(torch.utils.data.Subset(omniglot_train, list(range(len(omniglot_train)))))
        tasks_test.append(torch.utils.data.Subset(omniglot_test, list(range(len(omniglot_test)))))
        # NOTE: we rely on Avalanche's scenario to apply the transform lazily.
        tasks_train[-1].dataset.transform = tfm  # type: ignore
        tasks_test[-1].dataset.transform = tfm   # type: ignore

    return NCScenario(train_datasets=tasks_train, test_datasets=tasks_test, task_labels=False)
