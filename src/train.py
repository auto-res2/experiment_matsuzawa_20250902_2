"""
train.py  –  model architectures, continual-learning strategy, and other
training-centric utilities extracted from the original monolithic script.
All heavy-lifting that happens inside the training loop should live here so
that it can be re-used by notebooks, hyper-parameter searchers, or other
front-ends without having to touch `main.py`.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_small,
    resnet18,
    vit_b_16,
    ViT_B_16_Weights,
)

# -----------------------------------------------------------------------------
#  Avalanche strategy base class
# -----------------------------------------------------------------------------
# NOTE:  Avalanche reorganised its internal package structure in recent
#        versions (>=0.5).  The supervised continual-learning strategies such
#        as ``Naive`` have been moved from
#        ``avalanche.training.strategies`` to ``avalanche.training.supervised``.
#        To keep backward-compatibility with older versions while also
#        supporting the latest release, we attempt the new import first and
#        silently fall back to the old path if necessary.
# -----------------------------------------------------------------------------
try:
    # >= 0.5
    from avalanche.training.supervised import Naive
except ModuleNotFoundError:  # pragma: no cover
    # <= 0.4
    from avalanche.training.strategies import Naive  # type: ignore

# =============================================================================
#   Count-Sketch Fisher (parameter-space regulariser)
# =============================================================================


class CountSketchFisher(nn.Module):
    """Compressed Fisher information tracker using Count-Sketch.
    One instance per weight tensor.  The sketch table is stored in FP32 while
    the hash/sign vectors are stored in int tensors to minimise space usage.
    """

    def __init__(self, n_buckets: int = 1024):
        super().__init__()
        self.n_buckets = n_buckets
        # --- hash & sign vectors (sampled once, kept fixed) ------------------
        self.register_buffer(
            "h", torch.randint(0, n_buckets, (1,), dtype=torch.int64)
        )
        self.register_buffer(
            "s", torch.randint(0, 2, (1,), dtype=torch.int8) * 2 - 1
        )
        # sketch table holds the running sum of squared gradients
        self.table = nn.Parameter(
            torch.zeros(n_buckets, dtype=torch.float32), requires_grad=False
        )

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def accumulate(self, grad: torch.Tensor) -> None:
        """Accumulate a squared-gradient proxy for the (diagonal) Fisher."""
        flat = grad.view(-1)
        self.table[self.h] += (self.s.float() * flat) ** 2

    def penalty(self, grad: torch.Tensor) -> torch.Tensor:
        """Return the Count-Sketch projection penalty term ‖g·Cs‖²."""
        flat = grad.view(-1)
        return ((self.s.float() * flat).sum()) ** 2

    # -------------------- bookkeeping for memory footprint -------------------
    def extra_memory(self) -> int:  # bytes
        return (
            self.table.nelement() * self.table.element_size()
            + self.h.nelement() * self.h.element_size()
            + self.s.nelement() * self.s.element_size()
        )


# =============================================================================
#   Product-Quantised Prototype Buffer (data-side memory)
# =============================================================================


class ProductQuantisedBuffer:
    """A simple Product-Quantisation (PQ) buffer that stores class prototypes
    in compressed uint8 code form.  Codebooks are shared across all tasks to
    keep overall memory constant.
    """

    def __init__(self, feat_dim: int, n_subvectors: int = 4, k: int = 256):
        self.feat_dim = feat_dim
        self.n_subvec = n_subvectors
        self.sub_dim = feat_dim // n_subvectors
        self.k = k
        # create one code-book per sub-vector
        self.codebooks = [torch.zeros(k, self.sub_dim) for _ in range(n_subvectors)]
        # codes are stored per class:  class_id -> List[np.ndarray]
        self.codes: Dict[int, List[torch.ByteTensor]] = {}

    # ---------------------------------------------------------------------
    def add_prototypes(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """Add feature prototypes (one mean per class in the minibatch)."""
        feats_np = feats.cpu().numpy()
        labels_np = labels.cpu().numpy()
        for cls in set(labels_np.tolist()):
            idx = (labels_np == cls).nonzero()[0]
            proto = feats_np[idx].mean(axis=0, keepdims=True)  # 1×D
            codes_per_sub = []
            for s in range(self.n_subvec):
                chunk = proto[:, s * self.sub_dim : (s + 1) * self.sub_dim]
                cb = self.codebooks[s]
                # ------- populate or update the sub-codebook ---------------
                if (cb.abs().sum(dim=1) == 0).any():
                    empty_slot = (cb.abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0][0]
                    cb[empty_slot] = torch.from_numpy(chunk.squeeze()).float()
                    code_idx = empty_slot.item()
                else:
                    dists = (cb - torch.from_numpy(chunk)).pow(2).sum(dim=1)
                    code_idx = int(dists.argmin())
                    cb[code_idx] = 0.9 * cb[code_idx] + 0.1 * torch.from_numpy(
                        chunk.squeeze()
                    )
                codes_per_sub.append(torch.tensor(code_idx, dtype=torch.uint8))
            self.codes.setdefault(int(cls), []).append(torch.stack(codes_per_sub))

    # ---------------------------------------------------------------------
    def sample(self, n_per_class: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, labels = [], []
        for cls, code_list in self.codes.items():
            if not code_list:
                continue
            for code_vec in random.sample(code_list, k=min(n_per_class, len(code_list))):
                subv = []
                for s, code in enumerate(code_vec):
                    subv.append(self.codebooks[s][int(code)].unsqueeze(0))
                feats.append(torch.cat(subv, dim=1))
                labels.append(cls)
        if not feats:
            return torch.empty(0), torch.empty(0)
        return torch.cat(feats).float(), torch.tensor(labels)

    # ---------------------------------------------------------------------
    def memory_bytes(self) -> int:
        bytes_codebooks = sum(cb.nelement() * cb.element_size() for cb in self.codebooks)
        bytes_codes = sum(len(lst) * self.n_subvec for lst in self.codes.values())  # uint8 each
        return bytes_codebooks + bytes_codes


# =============================================================================
#   Lightweight Decoders (used to reconstruct images from PQ codes)
# =============================================================================


class DecoderCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3),
        )

    def forward(self, z):
        z = z.view(z.size(0), 192, 1, 1)
        return torch.sigmoid(self.net(z))


class DecoderOmniglot(nn.Module):
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, 784)
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z)).view(-1, 1, 28, 28)


# =============================================================================
#   Backbone Factory
# =============================================================================


def build_backbone(name: str):
    """Return (backbone_without_classifier, feature_dimension)."""
    if name == "resnet18":
        model = resnet18(weights=None)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim
    if name == "mobilenetv3":
        model = mobilenet_v3_small(weights=None)
        feat_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feat_dim
    if name == "vit_tiny":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        feat_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, feat_dim
    raise ValueError(f"Unknown backbone identifier: {name}")


# =============================================================================
#   SQM Continual-Learning Strategy (built on Avalanche's Naive)
# =============================================================================


class SQMStrategy(Naive):
    """Wrapper that augments Avalanche's `Naive` strategy with:
    1) Prototype replay via a PQ buffer,
    2) Count-Sketch Fisher regularisation,
    3) Gradient sparsification.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        feat_dim: int,
        lam: float = 3e-2,
        pq_clusters: int = 4,
        drop_ratio: float = 0.8,
        device: str | torch.device = "cuda",
    ):
        super().__init__(model, optimizer, criterion, device=device)
        self.lam = lam
        self.drop_ratio = drop_ratio
        self.pq_memory = ProductQuantisedBuffer(feat_dim, n_subvectors=4)
        self.decoder = DecoderCIFAR().to(device)
        # one Count-Sketch per trainable parameter
        self.sketches: Dict[nn.Parameter, CountSketchFisher] = {
            p: CountSketchFisher() for p in model.parameters() if p.requires_grad
        }
        # cache for per-batch features
        self._current_features: torch.Tensor | None = None

    # ---------------------------------------------------------------------
    def _after_forward(self, **kwargs):  # Avalanche hook
        # store penultimate features for later PQ update.  Depending on the
        # model, features may be available in kwargs; fall back to mb_x.
        self._current_features = kwargs.get("out", self.mb_x.detach())

    # ---------------------------------------------------------------------
    def _after_backward(self):  # Avalanche hook
        penalty = torch.zeros((), device=self.device)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            # ---------------------- gradient sparsification ---------------
            k = int(p.grad.numel() * (1 - self.drop_ratio))
            if k > 0:
                vals, idx = torch.topk(p.grad.abs().flatten(), k)
                mask = torch.zeros_like(p.grad.view(-1))
                mask[idx] = 1.0
                p.grad.mul_(mask.view_as(p.grad))
            # ---------------------- Count-Sketch penalty ------------------
            sketch = self.sketches[p]
            penalty = penalty + sketch.penalty(p.grad)
            sketch.accumulate(p.grad)
        # inject regularisation term
        self.loss = self.loss + self.lam * penalty

    # ---------------------------------------------------------------------
    def _after_update(self, **kwargs):  # Avalanche hook
        if self._current_features is None:
            return
        self.pq_memory.add_prototypes(
            self._current_features.detach(), self.mb_y.detach()
        )

    # ---------------------------------------------------------------------
    def collect_replay(self, n_per_class: int = 1):
        """Sample prototypes and reconstruct approximate images for replay."""
        if self.pq_memory is None:
            return None, None
        feats, labels = self.pq_memory.sample(n_per_class)
        if feats.numel() == 0:
            return None, None
        imgs = self.decoder(feats.to(self.device))
        return imgs, labels.to(self.device)

    # ---------------------------------------------------------------------
    def extra_memory_bytes(self) -> int:
        dec_bytes = sum(p.nelement() * p.element_size() for p in self.decoder.parameters())
        cs_bytes = sum(sk.extra_memory() for sk in self.sketches.values())
        pq_bytes = 0 if self.pq_memory is None else self.pq_memory.memory_bytes()
        return dec_bytes + cs_bytes + pq_bytes
