from __future__ import annotations
import json
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from .preprocess import (
    set_seed,
    prepare_external_files,
    load_planetoid_dataset,
)
from .train import ADRGNN, VanillaGCN, train_model

###############################################################################
#                    Depth-scaling stress-test (Experiment-1)                 #
###############################################################################

def run_experiment1(cfg: Dict) -> Dict:
    """Replicates the light-weight demo of the original monolithic script."""

    print("\n================ EXPERIMENT 1 – DEPTH-SCALING STRESS-TEST ================")
    print(
        f"Running on device {cfg['device']} with depths {cfg['depths']} for {cfg['epochs']} epochs.\n"
    )

    # make sure raw files are present (small and fast to fetch)
    prepare_external_files(cfg["dataset_root"])

    results: Dict[str, Dict[int, Dict[str, float]]] = {"ADR": {}, "GCN": {}}

    for depth in cfg["depths"]:
        for model_name in ["ADR", "GCN"]:
            accs: List[float] = []
            apsds: List[float] = []
            for seed in cfg["seeds"]:
                set_seed(seed)
                data, dataset = load_planetoid_dataset("Cora", cfg["dataset_root"], cfg["device"])

                if model_name == "ADR":
                    model = ADRGNN(dataset.num_features, cfg["hidden"], dataset.num_classes, depth).to(
                        cfg["device"]
                    )
                else:
                    model = VanillaGCN(dataset.num_features, cfg["hidden"], dataset.num_classes, depth).to(
                        cfg["device"]
                    )

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                crit = nn.CrossEntropyLoss()

                hist = train_model(
                    model,
                    data,
                    optimizer,
                    crit,
                    data.train_mask,
                    data.val_mask,
                    data.test_mask,
                    cfg["epochs"],
                    cfg["patience"],
                )

                accs.append(hist["test_acc"])
                apsds.append(hist["apsd"][-1] if hist["apsd"] else 0.0)

            results[model_name][depth] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "apsd_mean": float(np.mean(apsds)),
            }
            print(
                f"{model_name} depth {depth}: acc {results[model_name][depth]['acc_mean']:.2f} "+
                f"±{results[model_name][depth]['acc_std']:.2f} | APSD {results[model_name][depth]['apsd_mean']:.4f}"
            )

    _plot_depth_scaling(results, cfg)
    return results


###############################################################################
#                                   plots                                    #
###############################################################################

def _plot_depth_scaling(results: Dict, cfg: Dict):
    depths = cfg["depths"]
    adr_vals = [results["ADR"][d]["acc_mean"] for d in depths]
    gcn_vals = [results["GCN"][d]["acc_mean"] for d in depths]

    os.makedirs(cfg["fig_dir"], exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(depths, adr_vals, marker="o", label="ADR-GNN")
    plt.plot(depths, gcn_vals, marker="s", label="GCN")
    for x, y in zip(depths, adr_vals):
        plt.text(x, y + 0.3, f"{y:.1f}")
    for x, y in zip(depths, gcn_vals):
        plt.text(x, y - 2.5, f"{y:.1f}")
    plt.xlabel("Depth (layers)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Depth-Scaling Accuracy on Cora (demo run)")
    plt.legend()
    fname = os.path.join(cfg["fig_dir"], "accuracy_depth_scaling.pdf")
    try:
        plt.savefig(fname, bbox_inches="tight")
        print("Saved figure:", fname)
    except Exception as e:
        print("[Warning] Failed to save figure:", e)
    plt.close()
