"""
main.py
Entry-point orchestrating all experiments.
Run with  :   python -m src.main
"""
from __future__ import annotations
import time

from .evaluate import exp1_run, exp2_run, exp3_run


def main():
    tic = time.time()
    exp1_run()
    exp2_run()
    exp3_run()
    print(f"Done.  Total wall-clock time {time.time()-tic:.1f} s.")

if __name__ == "__main__":
    main()
