"""
main.py – top-level orchestration script (entry-point: `python -m src.main`)
The logic is a cleaned-up copy of the MASTER RUNNER section from the original
monolithic experiment file.  All heavy lifting is delegated to `train.run_single`.
"""
from __future__ import annotations
import time, math, json
from pathlib import Path
import numpy as np

from src.train import run_single
from src.metrics import paired_t_test
from src.viz import save_results_table

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#                      EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------------
EXP_LIST = [
    "exp1_waterbirds",   # Waterbirds-95, 100 epochs
    "exp1_celeba",       # CelebA Hair,  80 epochs
    "exp2_cars",         # Diff-Shift-Cars, 60 epochs
    "exp3_ninco",        # ImageNet-mini → NINCO/CIFAR-C, 30 epochs
]

DEFAULT_SEEDS = {
    "exp1_waterbirds": [0, 1, 2, 3, 4],
    "exp1_celeba"    : [0, 1, 2, 3, 4],
    "exp2_cars"      : [1, 2, 3, 4, 5],
    "exp3_ninco"     : [0, 1, 2],
}

# ---------------------------------------------------------------------------
#                           MASTER RUNNER
# ---------------------------------------------------------------------------

def run_all():
    results = {}
    for exp in EXP_LIST:
        methods = (
            ["erm", "irm", "fishr", "fourier", "autoacer", "cclidar", "groupdro_oracle"]
            if exp.startswith("exp1") else (
                ["erm", "cad_gan", "autoacer", "cclidar"] if exp.startswith("exp2") else
                ["erm", "no_ace", "no_gcdro", "no_hff", "cclidar"]
            )
        )

        for m in methods:
            res_per_seed = []
            for s in DEFAULT_SEEDS[exp]:
                res_per_seed.append(run_single(s, exp, m))

            # aggregate mean ± se
            res_arr = np.array(res_per_seed)
            mean, se = res_arr.mean(0), res_arr.std(0) / math.sqrt(len(res_arr))
            p = paired_t_test(res_arr, baseline="erm") if m != "erm" else None
            results.setdefault(exp, {})[m] = dict(mean=mean.tolist(), se=se.tolist(), p=p)

        # after each dataset create result table + PDF
        save_results_table(results[exp], FIG_DIR / f"results_{exp}.pdf")

    # ----------- dump master CSV -------------
    with open(ROOT / "results_master.csv", "w") as f:
        json.dump(results, f, indent=2)
    print("All experiments done.  Master results written → results_master.csv")


# ---------------------------------------------------------------------------
#                               ENTRY-POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()
    print("\n====================  CC-LiDAR CONSISTENCY SUITE  ====================")
    print(
        "This run executes all experiments exactly as specified (epochs, seeds, baselines).\n"
        "Expected wall-clock ≤ 3×24 h on a single T4.  Abort via CTRL-C to resume later."
    )
    run_all()
    print(f"Total pipeline time: {(time.time() - start) / 3600:.2f} h")
