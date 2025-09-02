"""src/main.py
Entry-point that orchestrates all experiments.
Run via:   python -m src.main
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from .evaluate import run_exp1

EXPERIMENTS = {
    "exp1": run_exp1,
    # "exp2": run_exp2,  # Place-holders for future experiments
}


def main() -> None:  # noqa: D401
    print("Elastic Feature-Sketching Replay – Reproducibility Suite")
    print("--------------------------------------------------------")

    if not torch.cuda.is_available():
        sys.exit("CUDA device not found – the experiments require an NVIDIA GPU.")

    out_root = Path(__file__).resolve().parent.parent / "outputs"

    for name, fn in EXPERIMENTS.items():
        print(f"\n>>> Running {name}")
        try:
            fn(out_root / name)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Experiment {name} failed with error: {e}")
            raise  # Re-raise for visibility

    print("\nAll experiments completed ✔︎")


if __name__ == "__main__":
    main()
