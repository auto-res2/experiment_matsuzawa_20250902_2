from __future__ import annotations
import json
import time
import os
from pathlib import Path
from typing import Dict

import torch
import yaml

from .evaluate import run_experiment1

###############################################################################
#                                0.  Helpers                                 #
###############################################################################

def _load_config() -> Dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config yaml not found at {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # resolve automatic device selection
    if cfg.get("device", "auto") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # create output dirs if absent
    os.makedirs(cfg["dataset_root"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)
    return cfg

###############################################################################
#                               1.  Main run                                 #
###############################################################################

def main():
    cfg = _load_config()
    t0 = time.time()
    print("Running ADR-GNN refactored project on", cfg["device"])

    exp1_results = run_experiment1(cfg)

    print("\n==== EXPERIMENT 1 RESULTS (JSON) ====")
    print(json.dumps(exp1_results, indent=2))

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
