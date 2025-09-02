"""src/main.py
---------------------------------------------------------------------
Entry-point that orchestrates the complete experimental workflow.
Invoke via:

    $ python -m src.main
"""

from __future__ import annotations

import time
import numpy as np
import torch

from .evaluate import run_experiment_1

# -------------------------------------------------------------------
#  Reproducibility – deterministic seeds & CuDNN settings
# -------------------------------------------------------------------

def _set_seeds(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    _set_seeds(0)
    tic = time.time()
    run_experiment_1()
    toc = time.time()
    print(f"\nTotal wall-clock time: {(toc - tic)/60:.1f} min")


if __name__ == "__main__":
    main()
