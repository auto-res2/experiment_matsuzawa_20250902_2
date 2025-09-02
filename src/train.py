"""
train.py – handles one complete training run (dataset, method, seed)
This file is almost a verbatim extraction of the training-related logic that
previously lived in the monolithic script.  Only minimal modifications were
performed:
  • Imports now point to the new, refactored modules (e.g. src.preprocess).
  • A FIG_DIR is created (if missing) to avoid runtime errors when saving
    figures.
  • The code is wrapped in a dedicated public function `run_single` that can
    be imported by other scripts (main.py).
No new modelling logic has been introduced.
"""
from __future__ import annotations
import math, time, json
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# ----------------------------------------------------------------------------
# local imports – kept identical to the original single-file implementation
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from src.preprocess import get_dataloaders, OracleGroupGuard
from src.methods import build_method
from src.metrics import MetricLogger, paired_t_test
from src.viz import save_loss_curves
from src.utils import set_seed, timeit, save_checkpoint, WandBLogger

__all__ = ["run_single"]

# ---------------------------------------------------------------------------
#                         SINGLE-RUN FUNCTION
# ---------------------------------------------------------------------------

@timeit
def run_single(seed: int, exp_name: str, method_name: str) -> Dict[str, Iterable]:
    """Run ONE (dataset, method, seed) experiment – full training + test eval.

    The body of this function is taken directly from the original script,
    with only cosmetic changes (FIG_DIR fix and updated imports).
    """
    set_seed(seed)
    print(f"\n========= RUN {exp_name} — {method_name} — seed {seed} =========")

    # ---------- dataloaders & oracle-guard ----------
    train_dl, val_dl, test_dl, n_classes = get_dataloaders(exp_name, batch_sz=64)
    guard = OracleGroupGuard(train_phase=True)  # will switch to False after train

    # ---------- build model & optimiser -------------
    model, optimiser, scheduler, cfg = build_method(method_name, n_classes, exp_name)
    scaler = GradScaler(enabled=cfg.fp16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------- loggers ----------
    meter = MetricLogger()
    wb = WandBLogger(exp_name, method_name, seed, cfg)

    # --------------- training loop ---------------
    total_epochs = cfg.epochs
    for epoch in range(1, total_epochs + 1):
        model.train(); guard.train_phase = True
        pbar = tqdm(train_dl, leave=False, desc=f"train e{epoch}/{total_epochs}")

        for batch in pbar:
            # forbid oracle group access during training
            assert "g" not in batch or not guard(), "[BUG] group labels used in train step!"
            x, y = batch["x"].to(device), batch["y"].to(device)
            with autocast(enabled=cfg.fp16):
                out = model(batch)  # each method returns dict with ‘loss’ & components
            scaler.scale(out["loss"]).backward()
            scaler.step(optimiser)
            scaler.update(); optimiser.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=out["loss"].item())
            meter.update_train(out, batch)
        scheduler.step()

        # -------------- validation --------------
        model.eval(); guard.train_phase = False
        with torch.no_grad():
            for batch in val_dl:
                x, y, g = batch["x"].to(device), batch["y"].to(device), batch["g"].to(device)
                logits = model.forward_backbone(x)
                meter.update_val(logits, y, g)

        wb.log_epoch(epoch, meter)
        save_checkpoint(model, cfg, seed, exp_name, method_name, epoch, meter)

    # ------------- final test ---------------
    print("Evaluating on TEST …")
    test_metrics = meter.eval_split(model, test_dl, device)
    wb.log_final(test_metrics)

    # ------------- figures & tables ------------
    save_loss_curves(meter.loss_history, FIG_DIR / f"training_loss_{method_name}_{exp_name}.pdf")
    return test_metrics
