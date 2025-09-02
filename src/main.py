"""src/main.py
Main entry-point that orchestrates training & evaluation.
Run with:
    python -m src.main
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch

from .preprocess import load_dataset, set_seed
from .train import ACuDiN, make_baseline, train_step, ds_num_classes, DEVICE
from .evaluate import evaluate_model

# -----------------------------------------------------------------------------
#  Directories for outputs
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"
for d in [LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
#  Compact experiment runner (quick CI sanity)
# -----------------------------------------------------------------------------
class ExperimentRunner:
    """Runs the depth-robustness experiment (quick mode by default)."""

    def __init__(self, full_run: bool = False):
        self.full_run = full_run or (os.getenv("FULL_RUN", "0") == "1")
        self.results: Dict[str, Dict] = {}

    # ---------------------------------------------------------------------
    def experiment_depth(self):
        print("\n================ EXPERIMENT 1 – DEPTH-ROBUST ACCURACY & SMOOTHING ================")
        datasets = ["cora", "chameleon"] if not self.full_run else [
            "cora",
            "citeseer",
            "pubmed",
            "cornell",
            "texas",
            "wisconsin",
            "chameleon",
            "squirrel",
        ]
        depths = [2, 8, 32] if not self.full_run else [2, 4, 8, 16, 32, 64]
        baselines = ["gcn"]

        for dname in datasets:
            data = load_dataset(dname)
            masks = {
                "train": data.train_mask,
                "val": getattr(data, "val_mask", data.train_mask),
                "test": data.test_mask,
            }
            self.results[dname] = {}
            for depth in depths:
                print(f"\nDataset={dname}  Depth={depth}")
                # ---------------- ACuDiN ----------------
                model = ACuDiN(
                    data.num_features, hidden=64, out_dim=ds_num_classes(data), num_layers=depth
                ).to(DEVICE)
                opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
                t0 = time.time()
                epochs = 1 if not self.full_run else 300
                for _ in range(epochs):
                    train_step(model, data, opt, masks["train"])
                accs, rd, cd = evaluate_model(model, data, masks)
                self._log_result(dname, depth, "ACuDiN", accs["test"], rd, cd, time.time() - t0)

                # ---------------- baselines -------------
                for base in baselines:
                    model_b = make_baseline(
                        base, data.num_features, ds_num_classes(data), depth
                    ).to(DEVICE)
                    opt_b = torch.optim.AdamW(model_b.parameters(), lr=5e-4, weight_decay=1e-4)
                    t0_b = time.time()
                    for _ in range(epochs):
                        train_step(model_b, data, opt_b, masks["train"])
                    accs_b, rd_b, cd_b = evaluate_model(model_b, data, masks)
                    self._log_result(
                        dname, depth, base.upper(), accs_b["test"], rd_b, cd_b, time.time() - t0_b
                    )
        self._plot_depth_curves()

    # ---------------------------------------------------------------------
    def _log_result(self, dname, depth, model, acc, rd, cd, runtime):
        rec = {"accuracy": acc, "row_diff": rd, "col_diff": cd, "runtime": runtime}
        self.results[dname].setdefault(model, {})[depth] = rec
        print(
            f"  {model:<7} Acc={acc:6.3f}  RowDiff={rd:7.4f}  ColDiff={cd:7.4f}  Time={runtime:6.2f}s"
        )

    # ---------------------------------------------------------------------
    def _plot_depth_curves(self):
        for dname, models in self.results.items():
            plt.figure(figsize=(6, 4))
            for model_name, depth_dict in models.items():
                xs, ys = zip(*sorted(((k, v["accuracy"]) for k, v in depth_dict.items())))
                plt.plot(xs, ys, marker="o", label=model_name)
                for x, y in zip(xs, ys):
                    plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
            plt.xlabel("#Layers")
            plt.ylabel("Accuracy")
            plt.title(f"Depth-Robust Accuracy on {dname.capitalize()}")
            plt.xscale("log", basex=2)
            plt.gca().set_xticks(sorted(list({k for m in models.values() for k in m.keys()})))
            plt.legend()
            fname = FIG_DIR / f"accuracy_depth_{dname}.pdf"
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight", format="pdf")
            print(f"Saved figure: {fname}")
            plt.close()

    # ---------------------------------------------------------------------
    def run_all(self):
        self.experiment_depth()
        # Skipping other experiments in quick-mode; add here for full version.
        (LOG_DIR / "results.json").write_text(json.dumps(self.results, indent=2))
        print("\nAll experiments finished – numerical results written to logs/results.json")


# -----------------------------------------------------------------------------
#  Script entry point
# -----------------------------------------------------------------------------

def main():
    set_seed(0)
    runner = ExperimentRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
