"""src/main.py – entry-point orchestrating all experiments"""
import time
from pathlib import Path

import pytest

from .evaluate import run_exp1, run_exp2, run_exp3

# Mapping of experiment names to callables ------------------------------------
EXPS = {
    "byte_budget": run_exp1,
    "long_stream": run_exp2,
    "ablation": run_exp3,
}


def main():
    print("Elastic Feature-Sketching Replay – reproducibility suite\n")

    # optional quick mode for CI environments --------------------------------
    quick = bool(int(__import__("os").getenv("QUICK", "1")))
    print("Quick-mode:", quick)

    # lightweight internal tests (fail-fast) ----------------------------------
    print("Running unit tests …", end="", flush=True)
    code = pytest.main([str(Path(__file__).parent), "-q", "-k", "test_memory or test_sample_cost", "--maxfail=1"])
    assert code == 0, "unit tests failed"
    print("  OK ✔")

    for name, fn in EXPS.items():
        start = time.time()
        fn()
        print(f"{name} finished in {(time.time() - start) / 60:.1f} min\n")

    print("All experiments finished. Find outputs / figures under the outputs/ directory.")


if __name__ == "__main__":
    main()