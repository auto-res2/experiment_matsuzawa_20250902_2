"""
main.py
Entry point orchestrating the experiment from individual modules.
Run via:  python -m src.main
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm  # noqa: F401 – used in imported modules

from .train import SCaRINet, seed_everything, train_one_epoch
from .evaluate import evaluate, save_line_plot
from .preprocess import WaterbirdsWithCF

# -----------------------------------------------------------------------------
# Global constants & environment checks (Abort on failure)
# -----------------------------------------------------------------------------
DEFAULT_SEEDS = [0]
DATA_ENV_VARS = {
    "WATERBIRDS_ROOT": "Waterbirds images (WILDS) root directory",
    "WATERBIRDS_CF_ROOT": "Pre-generated Waterbirds counterfactual images root directory",
}

for env_var, human_msg in DATA_ENV_VARS.items():
    root = os.environ.get(env_var, "")
    if root == "" or not Path(root).exists():
        raise RuntimeError(
            f"[CONSISTENCY-CHECK] Environment variable {env_var} not set or path does not exist → {human_msg}."
        )
    # quick sanity: must contain at least one image file
    if len(list(Path(root).rglob("*.jpg"))) + len(list(Path(root).rglob("*.png"))) == 0:
        raise RuntimeError(f"[CONSISTENCY-CHECK] {env_var}='{root}' contains no images – aborting.")


# -----------------------------------------------------------------------------
#   Experiment 1 – SCaRI vs ERM on Waterbirds (ResNet-50, seed 0)
# -----------------------------------------------------------------------------

def run_experiment_1() -> None:
    print("\n================  Experiment 1 – Benchmark Robustness  ================")
    print("This run executes BOTH baseline ERM and our SCaRI method on Waterbirds with ResNet-50 backbone.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_table = []
    for method in ["erm", "scari"]:
        print(f"\n---------- Method: {method.upper()} ----------")
        seed_everything(DEFAULT_SEEDS[0])

        # ------------------ Data ------------------
        cf_root = Path(os.environ["WATERBIRDS_CF_ROOT"]) if method == "scari" else None
        train_ds = WaterbirdsWithCF("train", cf_root)
        val_ds = WaterbirdsWithCF("val", None)  # eval never needs CFs
        train_loader = DataLoader(
            train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
        )

        # ------------------ Model & optim ------------------
        model = SCaRINet("resnet50", n_classes=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)
        scaler = GradScaler()

        best_val_acc = -1.0
        history_acc, history_wg = [], []
        EPOCHS = 3 if os.environ.get("FAST_DEBUG", "0") == "1" else 30

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            if method == "scari":
                train_stats = train_one_epoch(model, train_loader, opt, scaler, device)
            else:
                # ---------- ERM: identical code path minus CF losses ----------
                model.train()
                ce_losses, accs = [], []
                for x, y, _meta, _ in tqdm(train_loader, desc="train", leave=False):
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast():
                        _feat, _proj, logits = model(x)
                        loss = torch.nn.functional.cross_entropy(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    ce_losses.append(loss.item())
                    accs.append((logits.argmax(1) == y).float().mean().item() * 100.0)
                train_stats = {
                    "train_loss": float(np.mean(ce_losses)),
                    "train_acc": float(np.mean(accs)),
                }

            val_stats = evaluate(model, val_loader, device)
            history_acc.append(val_stats["val_acc"])
            history_wg.append(val_stats["val_worst_group_acc"])
            print(
                f"Epoch {epoch:02d}/{EPOCHS}  |  tr-loss {train_stats['train_loss']:.3f}  "
                f"tr-acc {train_stats['train_acc']:.1f}  val-acc {val_stats['val_acc']:.1f}  "
                f"worst-grp {val_stats['val_worst_group_acc']:.1f}  (t {time.time()-t0:.1f}s)"
            )
            if val_stats["val_acc"] > best_val_acc:
                best_val_acc = val_stats["val_acc"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # --------------- save figure per method ---------------
        save_line_plot(
            list(range(1, EPOCHS + 1)),
            {"val_acc": history_acc, "worst_group": history_wg},
            title=f"{method.upper()} – Waterbirds Validation",
            ylab="Accuracy (%)",
            fname=f"waterbirds_{method}_accuracy",
        )

        results_table.append(
            {
                "method": method,
                "best_val_acc": best_val_acc,
                "last_val_worst_group": history_wg[-1],
            }
        )

    # ------------------- Print summary -------------------
    print("\n================  Experiment 1 – Numerical Summary  ================")
    print(json.dumps(results_table, indent=2))
    print(
        "Figures written: waterbirds_erm_accuracy.pdf, waterbirds_scari_accuracy.pdf"
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    run_experiment_1()


if __name__ == "__main__":  # pragma: no cover
    main()
