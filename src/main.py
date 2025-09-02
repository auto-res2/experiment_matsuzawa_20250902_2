"""src/main.py
Entry-point.  Run with:  python -m src.main
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import time
import json
from types import SimpleNamespace
from pathlib import Path
import textwrap

# ---------------------------------------------------------------------
# Early: ensure heavy dependencies are installed *before* other sub-modules
# ---------------------------------------------------------------------
REQUIRED_PIPS = [
    "torch",
    "torchvision",
    "torchaudio",
    "timm>=0.9.2",
    "diffusers>=0.19.0",
    "transformers>=4.33.0",
    "scikit-learn",
    "fvcore",
    "open_clip_torch",
    "pandas",
    "seaborn",
    "matplotlib",
    "tqdm",
    "requests",
    "numpy",
    "pillow",
]

def _ensure_pkg(pkg: str):
    name = pkg.split("==")[0].split(">=")[0].split("<=")[0]
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for _p in REQUIRED_PIPS:
    _ensure_pkg(_p)

# ---------------------------------------------------------------------
# Now safe to import internal modules that rely on those packages
# ---------------------------------------------------------------------
import torch
from torch.cuda.amp import GradScaler

from . import preprocess as prep
from .train import (
    build_backbone,
    CCLiDAR,
    train_one_epoch,
)
from .evaluate import evaluate, plot_training_curves

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = PROJECT_ROOT / "checkpoints"
FIG_ROOT = PROJECT_ROOT / "figs"
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
#  Experiment 1: Waterbirds demo
# ---------------------------------------------------------------------

def run_experiment1():
    print("\n=====================  Experiment 1  =====================")
    print(
        textwrap.dedent(
            """
        Zero-Annotation Worst-Group Robustness on Waterbirds & CelebA
        Goal: show that CC-LiDAR boosts worst-group accuracy without hurting ID accuracy.
        Backbone: ResNet-50 initialised with DINO weights.
        Baselines: ERM vs CC-LiDAR (full).
        """
        )
    )

    cfg = SimpleNamespace(
        epochs=3,  # use 3 epochs for quick demo
        batch_size=32,
        lr=3e-4,
        fp16=True,
        latent_h=224,
        latent_w=224,
        lambda_ce=1.0,
        lambda_hff=0.05,
        tau=0.7,
        strong_aug=prep.strong_aug,
    )

    # --------------------- Data ---------------------
    train_loader, val_loader, test_loader = prep.build_dataloaders("waterbirds", cfg.batch_size)

    # --------------------- Model & optimiser ---------------------
    backbone = build_backbone(num_classes=2)
    model = CCLiDAR(backbone, cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=cfg.fp16)

    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_acc": [], "val_wg": []}

    for epoch in range(cfg.epochs):
        print(f"Epoch [{epoch+1}/{cfg.epochs}]")
        tloss = train_one_epoch(model, train_loader, opt, scaler, cfg)
        acc_val, wg_val = evaluate(model, val_loader)
        print(f"Val   acc={acc_val:.2f}  worst-group={wg_val:.2f}  train-loss={tloss:.3f}")

        history["train_loss"].append(tloss)
        history["val_acc"].append(acc_val)
        history["val_wg"].append(wg_val)

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed to improve.")

    ckpt_path = CKPT_ROOT / "exp1_best_cc_lidar.pt"
    torch.save(best_state, ckpt_path)
    print(f"Saved best checkpoint to {ckpt_path.relative_to(PROJECT_ROOT)}")

    # --------------------- Test ---------------------
    model.load_state_dict(best_state)
    test_acc, test_wg = evaluate(model, test_loader)
    print(f"TEST  acc={test_acc:.2f}  worst-group={test_wg:.2f}")

    # --------------------- Figures ---------------------
    fig_path = FIG_ROOT / "training_curves_exp1.pdf"
    plot_training_curves(history, fig_path)
    print(f"Saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    # --------------------- Numeric JSON ---------------------
    print("\nExperiment 1 results")
    print(
        json.dumps(
            {
                "val_best_acc": best_val_acc,
                "test_acc": test_acc,
                "test_worst_group_acc": test_wg,
            },
            indent=2,
        )
    )

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()
    run_experiment1()
    end = time.time()
    print(f"Total run-time: {(end - start) / 60:.1f} min")
