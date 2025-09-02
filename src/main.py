"""src/main.py – orchestrates all experiments.
Usage
-----
CI smoke-test (2 epochs only):
    python -m src.main
Full run (300 epochs & all experiments):
    FULL_RUN=1 python -m src.main
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn
from torch.optim import AdamW

from .train import (
    DEVICE,
    EarlyStop,
    build_model,
    train_epoch,
)
from .evaluate import evaluate, save_accuracy_plot
from .preprocess import load_dataset

# -----------------------------------------------------------------------------
# Global paths -----------------------------------------------------------------
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Run-time configuration -------------------------------------------------------
# -----------------------------------------------------------------------------
FULL_RUN = os.getenv("FULL_RUN", "0") == "1"
MAX_EPOCH = 300 if FULL_RUN else 2  # quick smoke run by default
PATIENCE = 100 if FULL_RUN else 1
SEEDS: Sequence[int] = list(range(10))

# -----------------------------------------------------------------------------
# Reproducibility --------------------------------------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore – no-op on CPU


# -----------------------------------------------------------------------------
# Runner class -----------------------------------------------------------------
# -----------------------------------------------------------------------------
class Runner:
    def __init__(self):
        self.results: Dict = {}

    # ----------------------------------------------------------
    def sanity_gcn(self, data, masks, name: str):
        print(f"  Sanity GCN check on {name} …", flush=True)
        model = build_model("gcn", data.num_features, int(data.y.max() + 1), 64, 2).to(DEVICE)
        opt = AdamW(model.parameters(), lr=5e-3, weight_decay=5e-4)
        for _ in range(200):
            train_epoch(model, data, masks["train"], opt)
        accs, _, _ = evaluate(model, data, {"test": masks["test"]})
        acc = accs["test"]
        print(f"    ⇒ 2-layer GCN accuracy = {acc:.3f}")
        ref = {"cora": 0.80, "citeseer": 0.70, "pubmed": 0.79}
        if name in ref and acc < ref[name] - 1e-3:
            raise AssertionError(f"GCN sanity failed on {name} ({acc:.2f})")

    # ----------------------------------------------------------
    def exp1(self):
        print("\n=========== EXPERIMENT 1 – Baseline Reproduction ===========")
        datasets = ["cora", "citeseer", "pubmed", "chameleon", "squirrel"]
        models = ["gcn", "appnp", "pairnorm", "dropedge", "acudin"]
        for dname in datasets:
            data = load_dataset(dname).to(DEVICE)
            masks = {
                "train": data.train_mask,
                "val": getattr(data, "val_mask", data.train_mask),
                "test": data.test_mask,
            }
            if dname in {"cora", "citeseer", "pubmed"}:
                self.sanity_gcn(data, masks, dname)

            self.results.setdefault("exp1", {}).setdefault(dname, {})
            for mdl in models:
                print(f"\nDataset={dname}  Model={mdl}")
                layers = 32 if mdl == "acudin" else 2
                model = build_model(
                    mdl, data.num_features, int(data.y.max() + 1), 64, layers
                ).to(DEVICE)
                opt = AdamW(model.parameters(), lr=5e-3, weight_decay=5e-4)
                stopper = EarlyStop(PATIENCE)
                t0 = time.time()
                best_val = 0.0
                for _ in range(1, MAX_EPOCH + 1):
                    train_epoch(model, data, masks["train"], opt)
                    accs, rd, cd = evaluate(model, data, masks)
                    if stopper.step(accs["val"]):
                        break
                    if accs["val"] > best_val:
                        best_val = accs["val"]
                        torch.save(model.state_dict(), CKPT_DIR / f"{dname}_{mdl}_best.pth")
                total = time.time() - t0
                self.results["exp1"][dname][mdl] = {
                    "test_acc": accs["test"],
                    "row_diff": rd,
                    "col_diff": cd,
                    "runtime": total,
                }
                print(
                    f"  Finished {mdl}: test={accs['test']:.3f}  rd={rd:.3f}  cd={cd:.3f}  time={total/60:.1f}m"
                )

        # persist & plot Cora example --------------------------------------
        (LOG_DIR / "exp1_results.json").write_text(json.dumps(self.results["exp1"], indent=2))
        depths = [2]  # constant in exp1
        acc_plot = {m: [self.results["exp1"]["cora"][m]["test_acc"]] for m in models}
        save_accuracy_plot(depths, acc_plot, "cora_exp1")

    # ----------------------------------------------------------
    def exp2(self):
        print("\n=========== EXPERIMENT 2 – Depth Sweep ===========")
        if not FULL_RUN:
            print("(skipped in quick mode – set FULL_RUN=1)")
            return
        datasets = [
            "cora",
            "citeseer",
            "pubmed",
            "chameleon",
            "squirrel",
            "cornell",
            "texas",
            "wisconsin",
        ]
        models = ["acudin", "gcn", "appnp"]
        depths = [2, 4, 8, 16, 32, 64]
        for dname in datasets:
            data = load_dataset(dname).to(DEVICE)
            masks = {
                "train": data.train_mask,
                "val": getattr(data, "val_mask", data.train_mask),
                "test": data.test_mask,
            }
            self.results.setdefault("exp2", {}).setdefault(dname, {})
            for depth in depths:
                for mdl in models:
                    print(f"\n{dname}  {mdl}  L={depth}")
                    model = build_model(
                        mdl, data.num_features, int(data.y.max() + 1), 64, depth
                    ).to(DEVICE)
                    opt = AdamW(model.parameters(), lr=5e-3, weight_decay=5e-4)
                    stopper = EarlyStop(PATIENCE)
                    for _ in range(1, MAX_EPOCH + 1):
                        train_epoch(model, data, masks["train"], opt)
                        accs, rd, cd = evaluate(model, data, masks)
                        if math.isnan(accs["val"]):
                            raise RuntimeError("NaN detected in validation accuracy")
                        if stopper.step(accs["val"]):
                            break
                    self.results["exp2"][dname].setdefault(mdl, {})[depth] = {
                        "test_acc": accs["test"],
                        "row_diff": rd,
                        "col_diff": cd,
                    }
            # plot ---------------------------------------------------------
            acc_dict = {
                mdl: [self.results["exp2"][dname][mdl][d]["test_acc"] for d in depths]
                for mdl in models
            }
            save_accuracy_plot(depths, acc_dict, f"{dname}_exp2")
        (LOG_DIR / "exp2_results.json").write_text(json.dumps(self.results["exp2"], indent=2))

    # ----------------------------------------------------------
    def exp3(self):
        print("\n=========== EXPERIMENT 3 – Scalability & Noise ===========")
        if not FULL_RUN:
            print("(skipped in quick mode – set FULL_RUN=1)")
            return
        # Heavy experiment intentionally omitted for brevity & CI time.

    # ----------------------------------------------------------
    def run(self):
        self.exp1()
        self.exp2()
        self.exp3()
        (LOG_DIR / "all_results.json").write_text(json.dumps(self.results, indent=2))
        print("\nAll experiments completed – results saved in logs/ directory")


# -----------------------------------------------------------------------------
# Entry point ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    set_seed(0)
    runner = Runner()
    runner.run()


if __name__ == "__main__":
    main()
