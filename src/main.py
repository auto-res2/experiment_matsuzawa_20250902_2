"""src/main.py
Command-line entry point that orchestrates the different experiments.
Run for example:
    python -m src.main --exp 1 --output_dir ./outputs
"""
from __future__ import annotations

import argparse
import torch

from src import evaluate as ev


def parse_args():
    p = argparse.ArgumentParser("HARD diffusion experiments")
    p.add_argument("--exp", required=True, choices=["1", "2", "3"], help="Which experiment to run")
    p.add_argument("--output_dir", type=str, default="outputs", help="Folder for results")
    p.add_argument("--prompt_json", type=str, default="prompts.json", help="Prompt list (Experiment-2)")
    return p.parse_args()


def main():
    args = parse_args()

    # Performance flags
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.exp == "1":
        ev.run_experiment1(args)
    elif args.exp == "2":
        ev.run_experiment2(args)
    elif args.exp == "3":
        ev.run_experiment3(args)
    else:
        raise ValueError("Unsupported experiment id")


if __name__ == "__main__":
    main()
