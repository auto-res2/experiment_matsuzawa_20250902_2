```
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
from avalanche.benchmarks import NCScenario

# -----------------------------------------------------------------------------
#  Omniglot import helper
# -----------------------------------------------------------------------------
# Newer versions of Avalanche (\u22650.6) removed the internal OmniglotDataset
# wrapper.  We therefore try to import it first and gracefully fall back to
# ``torchvision.datasets.Omniglot`` if it is no longer available.
# -----------------------------------------------------------------------------
try:  # noqa: WPS501 –  intentional broad except for optional dependency
    from avalanche.benchmarks.datasets import OmniglotDataset as _Omniglot  # type: ignore

    def _get_omniglot(*, train: bool, download: bool):  # noqa: D401 – simple
        """Return an Omniglot dataset instance (Avalanche implementation).

        The Avalanche wrapper uses the same API as the legacy code (train / test
        boolean flag), hence we simply forward the arguments.
        """
        return _Omniglot(train=train, download=download)  # type: ignore[arg-type]

except Exception:  # pragma: no cover –  fall-back path, see comment above
    from torchvision.datasets import Omniglot as _Omniglot  # type: ignore

    def _get_omniglot(*, train: bool, download: bool):  # noqa: D401 – simple
        """Return an Omniglot dataset instance (torchvision implementation).

        The torchvision variant uses the ``background`` keyword instead of
        ``train``.  We map *train=True* \u2192 *background=True* for parity with
        the original code.
        """
        root = Path("./data/omniglot")
        root.mkdir(parents=True, exist_ok=True)
        # torchvision: background=True corresponds to the training split
        return _Omniglot(
            root=str(root),  # torchvision expects a string path
            background=train,
            download=download,
        )

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


# -----------------------------------------------------------------------------
#  Omniglot Rotation Scenario
# -----------------------------------------------------------------------------

def build_omniglot_rotation_scenario(num_tasks: int = 100):
    """Create a rotated-Omniglot *NCScenario*, one unique rotation per task."""
    # ------------------------------------------------------------------
    # 1. Load base datasets (train / test)
    # ------------------------------------------------------------------
    omniglot_train = _get_omniglot(train=True, download=True)
    omniglot_test = _get_omniglot(train=False, download=True)

    # ------------------------------------------------------------------
    # 2. Generate task-specific rotated subsets
    # ------------------------------------------------------------------
    tasks_train: List[torch.utils.data.Dataset] = []
    tasks_test: List[torch.utils.data.Dataset] = []

    n_train = len(omniglot_train)
    n_test = len(omniglot_test)

    for task_id in range(num_tasks):
        # Each task is defined by a fixed rotation angle
        rot_angle = task_id * (360.0 / num_tasks)
        rot = T.RandomRotation((rot_angle, rot_angle))
        tfm = T.Compose([rot, T.Resize((28, 28)), T.ToTensor()])

        # Create full-dataset subsets so that we can assign a transform *per task*
        train_subset = torch.utils.data.Subset(omniglot_train, list(range(n_train)))
        test_subset = torch.utils.data.Subset(omniglot_test, list(range(n_test)))

        # Lazily apply the transform by altering the parent dataset attribute
        # This mirrors the behaviour in the original code base
        train_subset.dataset.transform = tfm  # type: ignore[attr-defined]
        test_subset.dataset.transform = tfm   # type: ignore[attr-defined]

        tasks_train.append(train_subset)
        tasks_test.append(test_subset)

    # ------------------------------------------------------------------
    # 3. Wrap everything into an Avalanche *NCScenario*
    # ------------------------------------------------------------------
    return NCScenario(
        train_datasets=tasks_train,
        test_datasets=tasks_test,
        task_labels=False,
    )
```
