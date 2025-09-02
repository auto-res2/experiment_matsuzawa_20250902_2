"""main.py
Entry point that orchestrates the three synthetic experiments using the
refactored modules.  It can be invoked via:

    python -m src.main
"""

import time
from .evaluate import run_experiment_1, run_experiment_2, run_experiment_3, device


def main():
    start = time.time()
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    elapsed = time.time() - start
    print(f"All synthetic experiments finished in {elapsed:.1f} s on {device}.")


if __name__ == "__main__":
    main()
