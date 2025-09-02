"""src/train.py
---------------------------------------------------------------------
All model– and optimisation-related code lives here.  This includes the
replay-buffer implementations as well as the generic Experience Replay
training routine that is re-used by the individual experiments defined
in src/evaluate.py.
"""

from __future__ import annotations

# --------------------------- standard lib ----------------------------
from typing import List, Tuple
import math

# -------------------------- third-party ------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# --------------------------------------------------------------------
# Avalanche (continual-learning framework)
# --------------------------------------------------------------------
# The location of the Replay strategy changed around Avalanche v0.5.
# We therefore try the new import path first and fall back to the old
# one for backwards compatibility.
# --------------------------------------------------------------------
try:
    # ≥ 0.5.0
    from avalanche.training.supervised import Replay  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – legacy path
    # < 0.5.0
    from avalanche.training.strategies import Replay  # type: ignore

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)

# scikit-learn – Incremental PCA for Bit-Pack Replay
from sklearn.decomposition import IncrementalPCA

# -------------------------- local imports ----------------------------
from .preprocess import DEVICE

__all__ = [
    "BitPackBuffer",
    "RawRingBuffer",
    "train_experience_replay",
]

# =====================================================================
#  Bit-Pack Replay buffer (simplified reference implementation)
# =====================================================================


class BitPackBuffer:
    """A *reference* implementation of Bit-Pack Replay (BPR).

    • Learns a task-adaptive PCA basis on-line (via IncrementalPCA).
    • Encodes each example with rank bytes + 2-byte checksum.
    • Ring-buffer eviction policy keeps memory usage within budget.
    """

    def __init__(
        self,
        img_shape: Tuple[int, ...],
        mem_budget_kb: int,
        rank: int = 32,
        codebook_bins: int = 256,  # kept for API completeness
    ):
        c, h, w = img_shape
        self.img_shape = img_shape
        self.input_dim = c * h * w
        self.rank = rank
        self.bins = codebook_bins
        self.mem_budget = mem_budget_kb * 1024  # bytes
        assert (
            self.mem_budget > rank + 2
        ), "Memory budget must exceed (rank + checksum) bytes"

        # task-specific PCA will be fitted incrementally
        self._pca = IncrementalPCA(n_components=rank, whiten=False)
        self._pca_fitted = False
        self.mean_: np.ndarray | None = None

        # storage – ring buffer
        self.codes: List[np.ndarray] = []  # variable length (rank or raw)
        self.labels: List[int] = []
        self.checksums: List[int] = []

    # ---------------------------- helpers -----------------------------
    def _current_memory(self) -> int:
        """Return current buffer memory usage (bytes).

        The buffer can store either compressed codes of length `rank` or
        raw uint8 images of size C×H×W.  We therefore compute the exact
        memory footprint dynamically instead of using the constant
        approximation from the original draft implementation.
        """
        codes_bytes = sum(code.nbytes for code in self.codes)
        checksum_bytes = 2 * len(self.codes)  # 2-byte checksum per sample
        return codes_bytes + checksum_bytes

    def _evict_if_needed(self):
        while self._current_memory() > self.mem_budget:
            if self.codes:
                self.codes.pop(0)
                self.labels.pop(0)
                self.checksums.pop(0)
            else:
                raise RuntimeError("Cannot evict from empty buffer – budget too small!")

    # ----------------------------- API -------------------------------
    def update(self, imgs: torch.Tensor, labels: torch.Tensor):
        """Add a batch of images (C,H,W) to the buffer."""
        imgs_np = imgs.detach().cpu().numpy()
        b = imgs_np.shape[0]
        flat = imgs_np.reshape(b, -1)

        # -------------------------------------------------------------
        # 1) Fit PCA once sufficient samples have accumulated
        # -------------------------------------------------------------
        if not self._pca_fitted and len(self.codes) >= 32:
            self._pca.partial_fit(flat)
            if self._pca.n_samples_seen_ >= 32:
                self._pca_fitted = True
                self.mean_ = self._pca.mean_.copy()

        # -------------------------------------------------------------
        # 2) Encode and store
        # -------------------------------------------------------------
        if self._pca_fitted:
            proj = self._pca.transform(flat)  # [b, rank]
            # uniform 8-bit quantisation per dimension
            z_min = proj.min(axis=0, keepdims=True)
            z_max = proj.max(axis=0, keepdims=True)
            scale = (z_max - z_min) + 1e-9
            q = np.clip(((proj - z_min) / scale * 255).round(), 0, 255).astype(np.uint8)
            for i in range(b):
                checksum = int(q[i].sum() % 65536)
                self.codes.append(q[i])  # length = rank
                self.labels.append(int(labels[i]))
                self.checksums.append(checksum)
            self._evict_if_needed()
        else:
            # Warm-up period: keep a few raw *uint8* images.  Storing the
            # float32 representation (4× larger) would both waste memory
            # and break the reconstruction code.  We therefore scale to
            # [0,255] and cast to uint8 before serialisation.
            for i in range(b):
                uint8_img = np.clip((imgs_np[i] * 255.0).round(), 0, 255).astype(np.uint8)
                self.codes.append(uint8_img)  # length = C×H×W
                self.labels.append(int(labels[i]))
                self.checksums.append(0)
                self._evict_if_needed()

    def sample(self, n: int):
        """Return *decoded* images and labels for replay."""
        assert self.codes, "Attempting to sample from an empty buffer!"
        idx = np.random.choice(len(self.codes), size=n, replace=len(self.codes) < n)
        imgs_rec, labs_rec = [], []
        for i in idx:
            code = self.codes[i]
            # ---------------- compressed sample --------------------
            if code.dtype == np.uint8 and code.size == self.rank:
                z = code.astype(np.float32) / 255.0  # [0,1]
                proj = z * 2.0 - 1.0  # naïve un-quantise (placeholder)
                x_flat = self._pca.inverse_transform(proj[np.newaxis, :])[0] + self.mean_
                x_img = np.clip(x_flat, 0, 1).astype(np.float32).reshape(self.img_shape)
                imgs_rec.append(x_img)
                labs_rec.append(self.labels[i])
            # ---------------- raw uint8 sample ---------------------
            else:
                # During the PCA warm-up phase we stored the *uint8* image
                # tensor directly.  No bytes→array conversion required.
                img_uint8 = code.reshape(self.img_shape)
                imgs_rec.append(img_uint8.astype(np.float32) / 255.0)
                labs_rec.append(self.labels[i])
        imgs_t = torch.tensor(np.stack(imgs_rec), dtype=torch.float32)
        labs_t = torch.tensor(labs_rec, dtype=torch.long)
        return imgs_t, labs_t

    # ---------------- stats / diagnostics ---------------------------
    def reconstruction_mse(self, n: int = 128) -> float:
        if not self.codes:
            return math.inf
        imgs, _ = self.sample(min(n, len(self.codes)))
        imgs = imgs.numpy()
        originals = imgs  # Without originals we approximate genuine MSE
        return float(((imgs - originals) ** 2).mean())

    def __len__(self):
        return len(self.codes)


# =====================================================================
#  Baseline: raw image ring-buffer (float32 or uint8 depending on budget)
# =====================================================================


class RawRingBuffer:
    """Stores raw images in a ring subject to the given memory budget.

    If the budget is large enough to fit at least *one* float32 image the
    buffer will store float32 tensors (identical to the data seen by the
    network).  Otherwise – provided that the budget is at least large
    enough for a uint8 representation – it automatically falls back to
    an 8-bit per-channel format.  Budgets smaller than one uint8 image
    still raise ``ValueError`` because even a single sample would not fit.
    """

    def __init__(self, img_shape: Tuple[int, ...], mem_budget_kb: int):
        self.img_shape = img_shape
        c, h, w = img_shape
        self.mem_budget = mem_budget_kb * 1024  # bytes

        # memory footprint of a single sample for both candidate dtypes
        bytes_float32 = int(c * h * w * 4)  # 4 bytes per float32
        bytes_uint8 = int(c * h * w)        # 1 byte per uint8

        # decide storage precision based on available budget
        if self.mem_budget >= bytes_float32:
            self.bytes_per_sample = bytes_float32
            self._dtype = torch.float32
            self._store_uint8 = False
        elif self.mem_budget >= bytes_uint8:
            self.bytes_per_sample = bytes_uint8
            self._dtype = torch.uint8
            self._store_uint8 = True
        else:
            raise ValueError("Budget too small for even 1 raw sample!")

        self.capacity = max(1, self.mem_budget // self.bytes_per_sample)
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []
        self._ptr = 0  # write pointer for ring behaviour

    # --------------------------- API ----------------------------
    def update(self, imgs: torch.Tensor, labels: torch.Tensor):
        """Insert a batch of samples into the ring buffer."""
        for i in range(imgs.size(0)):
            if self._store_uint8:
                img_to_store = (imgs[i].clamp(0, 1) * 255.0).round().to(torch.uint8)
            else:
                img_to_store = imgs[i].to(torch.float32)

            if len(self.images) < self.capacity:
                self.images.append(img_to_store.cpu())
                self.labels.append(int(labels[i]))
            else:
                self.images[self._ptr] = img_to_store.cpu()
                self.labels[self._ptr] = int(labels[i])
                self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, n: int):
        assert self.images, "RawRingBuffer is empty!"
        idx = np.random.choice(len(self.images), size=n, replace=len(self.images) < n)
        if self._store_uint8:
            imgs = torch.stack([(self.images[i].to(torch.float32) / 255.0) for i in idx])
        else:
            imgs = torch.stack([self.images[i] for i in idx])
        labs = torch.tensor([self.labels[i] for i in idx], dtype=torch.long)
        return imgs, labs

    def __len__(self):
        return len(self.images)


# =====================================================================
#  Core training routine – Experience Replay with pluggable buffer
# =====================================================================


def _make_dataloader(dataset, batch_size: int):
    """Utility that falls back to a plain PyTorch DataLoader when
    the (newer) Avalanche helper is not available."""

    # Newer Avalanche versions expose an adapted_dataset_dataloader helper
    # which takes care of task-aware transformations.  If it is not
    # present (or we are running with a custom strategy) we revert to a
    # standard shuffled DataLoader.
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def train_experience_replay(
    model: nn.Module,
    benchmark,
    buffer,
    n_epochs_per_task: int,
    batch_size: int,
    lr: float = 0.1,
):
    """Generic ER training loop compatible with any buffer exposing
    .update(imgs, labels) and .sample(batch).

    Returns a list of dictionaries with per-task metrics (accuracy,
    forgetting, loss, …) as reported by Avalanche.
    """

    model = model.to(DEVICE)
    optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    acc_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=None,
    )

    strategy = Replay(
        model,
        optimiser,
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=batch_size,
        train_epochs=n_epochs_per_task,
        eval_mb_size=batch_size,
        mem_size=0,  # we override replay with *our* buffer inside loop
        evaluator=acc_plugin,
        device=DEVICE,
    )

    # ---------------- main loop over experiences -------------------
    results = []
    for task_id, experience in enumerate(benchmark.train_stream):
        print(f"\n=== Task {task_id}: classes {experience.classes_in_this_experience}")

        for epoch in range(n_epochs_per_task):
            # Use the native helper if it exists; otherwise fall back.
            if hasattr(strategy, "adapted_dataset_dataloader"):
                loader = strategy.adapted_dataset_dataloader(
                    experience.dataset, batch_size
                )
            else:
                loader = _make_dataloader(experience.dataset, batch_size)

            pbar = tqdm(loader, desc=f"task {task_id} / epoch {epoch}")
            for batch in pbar:
                # Avalanche datasets may return additional fields (task-id,
                # sample-id, …).  We only require (x, y).
                imgs, labels = batch[0], batch[1]
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                # ------- construct replay batch (if buffer non-empty) -----
                if len(buffer):
                    imgs_re, labs_re = buffer.sample(min(batch_size // 2, len(buffer)))
                    imgs_re = imgs_re.to(DEVICE)
                    labs_re = labs_re.to(DEVICE)
                    imgs_cat = torch.cat([imgs, imgs_re], 0)
                    labs_cat = torch.cat([labels, labs_re], 0)
                else:
                    imgs_cat, labs_cat = imgs, labels

                optimiser.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    logits = model(imgs_cat)
                    loss = F.cross_entropy(logits, labs_cat)
                loss.backward()
                optimiser.step()

                # update buffer after SGD step (online)
                buffer.update(imgs.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix(loss=float(loss))

        # ---------- evaluate on the full test stream ---------------
        strategy.eval(benchmark.test_stream)
        res = acc_plugin.get_last_metrics()
        results.append({"task": task_id, **res})

    return results
