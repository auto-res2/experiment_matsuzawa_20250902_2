"""src/train.py
Utility functions and model / training related code extracted from the original
monolithic script.
Because the published experiments are *evaluation* only (no optimisation loop),
this file mainly provides building blocks that are *shared* by the different
experiments such as random-seed control and the HARD wrapper modules.
"""
from __future__ import annotations

from typing import List
import random
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "set_seed",
    "SpatialUncertaintyHead",
    "HARDWrapper",
]

# -----------------------------------------------------------------------------
# Re-usable utility
# -----------------------------------------------------------------------------
SEED_LIST = [13, 17, 23]


def set_seed(seed: int) -> None:
    """Fully deterministic seeding for `random`, `numpy` and `torch`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model components –  Hierarchical Adaptive Refinement for Diffusion (HARD)
# -----------------------------------------------------------------------------
class SpatialUncertaintyHead(nn.Module):
    """Light-weight 1×1-conv based head that predicts an uncertainty map.
    Output range: [0,1]. Shape preserved (only channel → 1)."""

    def __init__(self, in_channels: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,H,W]
        return self.net(x)


class HARDWrapper(nn.Module):
    """Framework-agnostic HARD wrapper that turns a *list* of per-level denoisers
    (e.g. the three U-Nets of a DiT cascade) into an adaptive multi-grid solver.
    Each element of `sub_models` **must** implement the callable interface
    `latent_out = sub_model(latent_in, t)`.
    """

    def __init__(
        self,
        sub_models: List[nn.Module],
        taus: List[float],
        freeze_k: int = 2,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        assert len(sub_models) == len(taus), "Need one τ per level"
        self.levels = len(sub_models)
        self.sub_models = nn.ModuleList(sub_models)
        self.uncert_heads = nn.ModuleList(
            [SpatialUncertaintyHead().to(device, dtype=dtype) for _ in range(self.levels)]
        )
        self.taus = taus
        self.freeze_k = freeze_k

        # Internal state that will be re-used between timesteps
        self.register_buffer("frozen", torch.zeros(self.levels, dtype=torch.bool))
        self.register_buffer("below_tau_counter", torch.zeros(self.levels, dtype=torch.long))
        self.cached_pred: List[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _should_eval(self, lvl: int, u_map: torch.Tensor) -> bool:
        """Return *True* iff we decide to run the expensive denoiser at this level."""
        mean_u = u_map.mean().item()
        if mean_u < self.taus[lvl]:
            self.below_tau_counter[lvl] += 1
        else:
            self.below_tau_counter[lvl] = 0
        if self.below_tau_counter[lvl] >= self.freeze_k:
            self.frozen[lvl] = True
        return not self.frozen[lvl].item()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, latents: List[torch.Tensor], timesteps: List[int]):
        """Perform *one* HARD step across all resolution levels.

        Args
        ----
        latents   : list[Tensor]  – latent at each scale (coarse → fine)
        timesteps : list[int]     – global diffusion timestep for each level
        """

        if self.cached_pred is None:
            # First call – allocate cache
            self.cached_pred = [torch.zeros_like(l) for l in latents]

        outs, u_maps = [], []
        for lvl in range(self.levels):
            # 1) Already frozen → use cache, uncertainty 0.
            if self.frozen[lvl]:
                outs.append(self.cached_pred[lvl])
                u_maps.append(torch.zeros_like(latents[lvl][:, :1]))
                continue

            # 2) Predict uncertainty, decide whether to run the expensive block
            u_map = self.uncert_heads[lvl](latents[lvl])
            do_eval = self._should_eval(lvl, u_map)
            if do_eval:
                pred = self.sub_models[lvl](latents[lvl], timesteps[lvl])
                self.cached_pred[lvl] = pred
            else:
                pred = self.cached_pred[lvl]
            outs.append(pred)
            u_maps.append(u_map)
        return outs, u_maps
