"""
main.py – orchestration entry point
Launch with  :   python -m src.main
"""
from __future__ import annotations
import argparse

import torch

from .evaluate import offline_sanity, experiment1, experiment2, experiment3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", default=True, help="CI-friendly short run")
    args = ap.parse_args()

    print("Running continual-learning experiments with PyTorch …  CUDA:", torch.cuda.is_available())

    # quick deterministic sanity check ------------------------------------------------
    offline_sanity(epochs=1 if args.fast else 20)

    seeds = [111, 222] if args.fast else [111, 222, 333, 444, 555, 666, 777, 888, 999, 1010]
    experiment1(seeds, fast=args.fast)
    experiment2()
    experiment3()

    print("\nAll experiments finished – figures in ./figures/ .")


if __name__ == "__main__":
    main()
