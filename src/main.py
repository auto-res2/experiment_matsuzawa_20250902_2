"""src/main.py
Entry point – dispatches experiment selection to evaluation routines.
(unchanged)
"""
from __future__ import annotations
import argparse

from .evaluate import exp1, exp2, exp3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3], help="Which experiment to run")
    args = parser.parse_args()

    if args.exp == 1:
        exp1()
    elif args.exp == 2:
        exp2()
    else:
        exp3()
