"""src/main.py
Execution entry-point: orchestrates Experiment-1 demonstrating SCaRI on Waterbirds.
Run with:  python -m src.main
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from .preprocess import set_seed, WaterbirdsWildsWrapper, collate_with_optional_cf
from .train import SCaRINet, train_epoch
from .evaluate import evaluate, save_line_fig

# -----------------------------------------------------------------------------
#  Constants & environment checks
# -----------------------------------------------------------------------------
GLOBAL_SEED = 0
DEFAULT_BS = 128
DEFAULT_EPOCHS = 30


def _choose_hyperparams(synthetic: bool):
    """Return (arch, batch_size, epochs) depending on dataset availability."""
    if synthetic:
        # Keep the CI run lightweight
        return "resnet18", min(32, DEFAULT_BS), 2
    return "resnet50", DEFAULT_BS, DEFAULT_EPOCHS


def run_experiment_1(seed: int = GLOBAL_SEED) -> None:
    print("===== Experiment 1 – Standard-Benchmark Robustness Sweep =====")
    print(
        "Goal: Verify that SCaRI preserves ID accuracy and boosts worst-group and OOD accuracy without spurious annotations.\n"
    )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    #  Initialise dataset (real or synthetic)
    # ------------------------------------------------------------------
    cf_root_env = os.environ.get("WATERBIRDS_CF_ROOT")
    cf_root = Path(cf_root_env) if cf_root_env and Path(cf_root_env).exists() else None
    if cf_root is None:
        print("[WARN] Counterfactual directory not found – training without CF branch.")

    train_ds = WaterbirdsWildsWrapper("train", cf_root=cf_root)
    val_ds = WaterbirdsWildsWrapper("val", cf_root=None)  # No CFs during validation

    arch, BS, EPOCHS = _choose_hyperparams(train_ds.synthetic)
    num_workers = 0  # Robust setting for most CI runners

    train_loader = DataLoader(
        train_ds, batch_size=BS, shuffle=True, num_workers=num_workers, collate_fn=collate_with_optional_cf
    )
    val_loader = DataLoader(
        val_ds, batch_size=BS, shuffle=False, num_workers=num_workers, collate_fn=collate_with_optional_cf
    )

    # ------------------------------------------------------------------
    #  Model, optimiser, scaler
    # ------------------------------------------------------------------
    model = SCaRINet(arch, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scaler = GradScaler()

    # ------------------------------------------------------------------
    #  Training/validation loop
    # ------------------------------------------------------------------
    train_acc_hist, val_acc_hist = [], []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            cit_tau=0.2,
            eta=0.2,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"train-loss {tr_loss:.4f} | train-acc {tr_acc:.2f} | val-loss {val_loss:.4f} | val-acc {val_acc:.2f}"
        )
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # ------------------------------------------------------------------
    #  Plotting and logging
    # ------------------------------------------------------------------
    epochs_range = list(range(1, EPOCHS + 1))
    save_line_fig(
        epochs_range,
        {"train_acc": train_acc_hist, "val_acc": val_acc_hist},
        title="Training vs Validation Accuracy (Waterbirds)",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
        filename="accuracy_waterbirds_scari",
    )

    print("Numerical Results: (mean over epochs)")
    print(
        json.dumps(
            {"train_acc_mean": float(np.mean(train_acc_hist)), "val_acc_best": best_val_acc},
            indent=2,
        )
    )

    if best_state is not None:
        torch.save(best_state, "best_model_scari_waterbirds.pth")
        print("Best model checkpoint saved: best_model_scari_waterbirds.pth")

    print("Figures generated: accuracy_waterbirds_scari.pdf")


def main() -> None:  # noqa: D401
    """Entry point used by `python -m src.main`."""
    try:
        run_experiment_1()
    except Exception as exc:  # pragma: no cover
        print("Fatal error during Experiment 1:", file=sys.stderr)
        raise exc


if __name__ == "__main__":
    main()
