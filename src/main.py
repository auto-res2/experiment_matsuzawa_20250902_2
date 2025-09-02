"""src/main.py
Command-line entry point that orchestrates the different experiments.
Run for example:
    python -m src.main --exp 1 --output_dir ./outputs
"""
from __future__ import annotations

import argparse
import sys
import torch

from src import evaluate as ev


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.
    The experiment flag is *optional* so that importing / unit–testing this
    module does **not** immediately trigger the heavy experiment code.  If no
    experiment is specified we just print the help text and exit gracefully.
    """
    parser = argparse.ArgumentParser("HARD diffusion experiments", add_help=True)
    parser.add_argument(
        "--exp",
        choices=["1", "2", "3"],
        default=None,
        help="Which experiment to run (omit for a dry run)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Folder for results",
    )
    parser.add_argument(
        "--prompt_json",
        type=str,
        default="prompts.json",
        help="Prompt list (Experiment-2)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Light wrapper so that unit tests can inject their own argv list."""
    parser = _build_arg_parser()
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # If no experiment was given we only show the help message and exit.  This
    # makes it possible to import this module (or call it without arguments)
    # during automated testing without launching the compute-intensive code.
    if args.exp is None:
        _build_arg_parser().print_help(sys.stderr)
        return

    # Performance flags – only relevant when GPU is available.
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
        raise ValueError(f"Unsupported experiment id: {args.exp}")


if __name__ == "__main__":
    main()
