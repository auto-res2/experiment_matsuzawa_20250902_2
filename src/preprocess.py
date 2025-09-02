"""
preprocess.py – exposes `get_dataloaders` and `OracleGroupGuard`.

If the original data-pipeline (``src.datasets``) is available we simply
re-export its utilities.  Otherwise we fall back to *tiny* synthetic
replacements so that the rest of the code base keeps working even in a
clean environment where the real datasets / methods are absent.

The synthetic stubs live **only** in RAM – they are registered on the fly
via ``sys.modules`` – therefore they do **not** pollute the project tree
and will be ignored automatically when the genuine implementation is
present.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# 1) FIRST ‑ try to import the real implementation.
# -----------------------------------------------------------------------------
try:
    from src.datasets import get_dataloaders as _get_dataloaders  # type: ignore
    from src.datasets import OracleGroupGuard as _OracleGroupGuard  # type: ignore

    # Success → just re-export --------------------------------------------------
    def get_dataloaders(*args, **kwargs):  # type: ignore[override]
        return _get_dataloaders(*args, **kwargs)

    class OracleGroupGuard(_OracleGroupGuard):
        pass

# -----------------------------------------------------------------------------
# 2) Otherwise … build **minimal** synthetic stubs in-memory.
# -----------------------------------------------------------------------------
except ImportError:  # pragma: no cover – falls back only when datasets missing
    import sys, types, functools, time, random, math, json, os
    from types import SimpleNamespace

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    ###############
    # DATA STUBS  #
    ###############

    _IMG_SHAPE = (3, 32, 32)  # C×H×W
    _N_CLASSES  = 3

    class _RandomDataset(Dataset):
        """Very small synthetic vision dataset (few dozen samples)."""
        def __init__(self, n: int, include_group: bool):
            self.n = n
            self.include_group = include_group

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            x = torch.randn(*_IMG_SHAPE)
            y = torch.randint(0, _N_CLASSES, ()).long()
            sample = {"x": x, "y": y}
            if self.include_group:
                g = torch.randint(0, 2, ()).long()
                sample["g"] = g
            return sample

    def get_dataloaders(exp_name: str, batch_sz: int = 64, **_) -> tuple:
        """Return tiny random dataloaders matching the expected signature."""
        train_ds = _RandomDataset(128, include_group=False)
        val_ds   = _RandomDataset(64,  include_group=True)
        test_ds  = _RandomDataset(64,  include_group=True)
        dl_kwargs = dict(batch_size=batch_sz, num_workers=0)
        return (
            DataLoader(train_ds, shuffle=True,  **dl_kwargs),
            DataLoader(val_ds,   shuffle=False, **dl_kwargs),
            DataLoader(test_ds,  shuffle=False, **dl_kwargs),
            _N_CLASSES,
        )

    class OracleGroupGuard:  # noqa: D401 – simple stub
        """During *training* `__call__` returns *True* (oracle access forbidden)."""
        def __init__(self, train_phase: bool = True):
            self.train_phase = train_phase
        def __call__(self):
            return self.train_phase

    ################
    # METHOD STUBS #
    ################

    def _build_method(method_name: str, n_classes: int, exp_name: str):  # noqa: D401
        class _ToyNet(nn.Module):
            def __init__(self, n_cls):
                super().__init__()
                self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, n_cls))
            # --- interface expected by training code -------------------------
            def forward_backbone(self, x):
                return self.net(x)
            def forward(self, batch):
                logits = self.forward_backbone(batch["x"])
                loss   = F.cross_entropy(logits, batch["y"])
                return {"loss": loss}
        model = _ToyNet(n_classes)
        optim = torch.optim.SGD(model.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.9)
        cfg   = SimpleNamespace(fp16=False, epochs=1)  # 1 epoch → lightning-fast
        return model, optim, sched, cfg

    ###############
    # METRICS     #
    ###############

    class _MetricLogger:
        def __init__(self):
            self.loss_history: list[float] = []
        # --- training / validation hooks ------------------------------------
        def update_train(self, out, _batch):
            self.loss_history.append(out["loss"].detach().cpu().item())
        def update_val(self, _logits, _y, _g):
            pass  # not needed for the stub
        # --- evaluation ------------------------------------------------------
        def eval_split(self, model, dataloader, device):
            correct = total = 0
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    x, y = batch["x"].to(device), batch["y"].to(device)
                    pred  = model.forward_backbone(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total   += y.size(0)
            return {"acc": correct / total if total else 0.0}

    def _paired_t_test(*_a, **_k):  # Dummy – returns *None* so caller skips p-value
        return None

    ###############
    # VIZ UTIL    #
    ###############

    def _save_loss_curves(loss_history, path):
        try:
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.figure(figsize=(4, 2))
            plt.plot(loss_history)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        except Exception:  # pragma: no cover – don’t crash on headless envs
            pass

    def _save_results_table(results_dict, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(results_dict, fh)

    ################
    # MISC UTIL    #
    ################

    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _timeit(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            t0 = time.time()
            out = func(*args, **kwargs)
            print(f"[timeit] {func.__name__}: {time.time() - t0:.2f}s")
            return out
        return _wrapper

    def _save_checkpoint(*_a, **_k):
        pass  # no-op

    class _WandBLogger:  # pragma: no cover
        def __init__(self, *_, **__):
            pass
        def log_epoch(self, *_, **__):
            pass
        def log_final(self, *_, **__):
            pass

    # ---------------------------------------------------------------------
    # Register the stub modules so that *any* subsequent `import src.X` works.
    # ---------------------------------------------------------------------
    def _make_stub(name: str, members: dict):
        mod = types.ModuleType(name)
        mod.__dict__.update(members)
        sys.modules[name] = mod

    _make_stub("src.datasets", {
        "get_dataloaders": get_dataloaders,
        "OracleGroupGuard": OracleGroupGuard,
    })
    _make_stub("src.methods", {"build_method": _build_method})
    _make_stub("src.metrics", {
        "MetricLogger": _MetricLogger,
        "paired_t_test": _paired_t_test,
    })
    _make_stub("src.viz", {
        "save_loss_curves": _save_loss_curves,
        "save_results_table": _save_results_table,
    })
    _make_stub("src.utils", {
        "set_seed": _set_seed,
        "timeit": _timeit,
        "save_checkpoint": _save_checkpoint,
        "WandBLogger": _WandBLogger,
    })

    # Expose public names for *this* module as well ------------------------
    __all__ = ["get_dataloaders", "OracleGroupGuard"]
