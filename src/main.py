"""src/main.py
Entry point that orchestrates all experiments.  Run via

    python -m src.main
"""
import time

from src.train import set_seed
from src.evaluate import Experiment1, Experiment2, Experiment3


def main():
    set_seed(11)

    experiments = [Experiment1(), Experiment2(), Experiment3()]
    for exp in experiments:
        start = time.perf_counter()
        exp.run()
        dur = time.perf_counter() - start
        print(f"Completed {exp.NAME} in {dur:.1f} s\n")

    print("All experiments finished.  Figures are saved in ./figures/*.pdf")


if __name__ == "__main__":
    main()
