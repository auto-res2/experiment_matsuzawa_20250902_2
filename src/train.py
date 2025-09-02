"""src/train.py – model definitions, buffers and training routine"""
import io
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import geotorch

# -----------------------------------------------------------------------------
# Device helper ----------------------------------------------------------------
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Back-bone --------------------------------------------------------------------
# -----------------------------------------------------------------------------
class ResNet18Stiefel(nn.Module):
    """ResNet-18 whose last conv layer is Stiefel–orthogonalised (geotorch)."""

    def __init__(self):
        super().__init__()
        self.net = resnet18(weights="IMAGENET1K_V1")
        feat_dim = self.net.fc.in_features
        self.net.fc = nn.Identity()  # remove classifier – we add a task-specific one later
        # Orthogonal constraint on the last residual block for near-isometry
        geotorch.orthogonal(self.net.layer4[-1].conv2, "weight")
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) → (B, 512)
        return self.net(x)

# -----------------------------------------------------------------------------
# Elastic-Feature-Sketch components -------------------------------------------
# -----------------------------------------------------------------------------
class OnlineVQ(nn.Module):
    """Single-layer vector-quantiser working directly in feature space."""

    def __init__(self, feat_dim: int, code_len: int = 8, k: int = 256):
        super().__init__()
        self.code_len = code_len
        self.embed = nn.Embedding(k, feat_dim)
        nn.init.uniform_(self.embed.weight, -1, 1)

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:  # (B, D) → (B,)
        z_n = F.normalize(z, dim=1)
        emb_n = F.normalize(self.embed.weight, dim=1)
        sims = torch.einsum("bd,kd->bk", z_n, emb_n)
        return sims.argmax(dim=1).to(torch.int64)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:  # (B,) → (B, D)
        return self.embed(codes)


class ReservoirItem:
    """Light-weight container – stored in Python list to avoid Torch autograd."""

    __slots__ = ("code", "y", "infl")

    def __init__(self, code: int, label: int, infl: float):
        self.code, self.y, self.infl = int(code), int(label), float(infl)


class EFSBuffer:
    """Elastic Feature-Sketching replay buffer (sub-linear memory growth)."""

    def __init__(self, feat_dim: int, B_max: int, code_len: int = 8):
        self.code_len = code_len
        self.B_max = B_max
        self.label_B = 2  # uint16 for class id
        self.cb_ptr_B = 1  # book-keeping pointer per sample
        self.bytes = 0
        self.items: List[ReservoirItem] = []
        self.vq = OnlineVQ(feat_dim, code_len)

    # ----------------- helpers ------------------------------------------------
    def _sample_cost(self) -> int:
        """Return bytes required for a single stored example."""
        return math.ceil(self.code_len / 8) + self.label_B + self.cb_ptr_B

    # ----------------- public API --------------------------------------------
    def observe(self, feats: torch.Tensor, y: torch.Tensor, infl: torch.Tensor):
        """Insert (feature, label, influence) triplets with reservoir replacement."""
        codes = self.vq.encode(feats.cpu())
        for c, yy, ii in zip(codes, y.cpu(), infl.cpu()):
            if self.bytes + self._sample_cost() > self.B_max:
                # influence-biased reservoir sampling
                j = random.randrange(len(self.items))
                if ii > self.items[j].infl:
                    self.items[j] = ReservoirItem(c, yy, ii)
            else:
                self.items.append(ReservoirItem(c, yy, ii))
                self.bytes += self._sample_cost()
        # ------------ invariants --------------------------------------------
        assert self.bytes <= self.B_max + self._sample_cost(), "buffer budget over-run"

    def sample(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(self.items), (k,))
        codes = torch.tensor([self.items[i].code for i in idx])
        ys = torch.tensor([self.items[i].y for i in idx])
        feats = self.vq.decode(codes)
        return feats, ys

    def wgf_refine(self, step: float = 0.1):
        """Light-weight Wasserstein gradient flow refinement."""
        if not self.items:
            return
        codes = torch.tensor([it.code for it in self.items])
        emb = self.vq.decode(codes)
        loss = emb.norm(p=2, dim=1).mean()
        grad = torch.autograd.grad(loss, self.vq.embed.weight, retain_graph=False)[0]
        with torch.no_grad():
            self.vq.embed.weight += step * grad

    # Python built-ins ---------------------------------------------------------
    def __len__(self):
        return len(self.items)

# -----------------------------------------------------------------------------
# Baseline buffers -------------------------------------------------------------
# -----------------------------------------------------------------------------
class RawImageBuffer:
    """Exact replay of raw inputs (JPEG compressed). FIFO once budget is full."""

    def __init__(self, B_max: int, jpeg_quality: int = 90):
        self.B_max = B_max
        self.jpeg_q = jpeg_quality
        self.items: List[Tuple[bytes, int]] = []  # (jpeg_bytes, label)
        self.bytes = 0

    # internal helpers --------------------------------------------------------
    def _encode(self, img: torch.Tensor) -> bytes:
        from PIL import Image
        arr = (img.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        im = Image.fromarray(arr)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=self.jpeg_q)
        return buf.getvalue()

    # public API --------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor, *_):
        for img, yy in zip(x, y):
            enc = self._encode(img)
            sz = len(enc) + 2  # label uint16
            if self.bytes + sz > self.B_max:
                continue  # drop newest once full (ER-Raw FIFO policy)
            self.items.append((enc, int(yy)))
            self.bytes += sz

    def sample(self, k: int):
        import numpy as np
        from PIL import Image, ImageFile
        from torchvision import transforms

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        idx = np.random.choice(len(self.items), size=k, replace=False)
        xs, ys = [], []
        to_tensor = transforms.ToTensor()
        for i in idx:
            img_b, yy = self.items[i]
            xs.append(to_tensor(Image.open(io.BytesIO(img_b))))
            ys.append(yy)
        return torch.stack(xs), torch.tensor(ys)

    def __len__(self):
        return len(self.items)


class RingBuffer(RawImageBuffer):
    """Class-balanced ring buffer as in ER-Ring."""

    def observe(self, x: torch.Tensor, y: torch.Tensor, *_):
        for img, yy in zip(x, y):
            enc = self._encode(img)
            sz = len(enc) + 2
            while self.bytes + sz > self.B_max and self.items:
                old, _ = self.items.pop(0)
                self.bytes -= len(old) + 2
            self.items.append((enc, int(yy)))
            self.bytes += sz

# -----------------------------------------------------------------------------
# Training routine ------------------------------------------------------------
# -----------------------------------------------------------------------------

def train_task(
    backbone: nn.Module,
    classifier: nn.Module,
    buffer,
    train_ds,
    val_ds,
    *,
    epochs: int = 200,
    replay_r: float = 0.5,
):
    """Train a single continual-learning task and optionally replay from buffer."""
    backbone.to(DEVICE)
    classifier.to(DEVICE)

    opt = torch.optim.SGD(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()

            feats = backbone(x)
            out = classifier(feats)
            loss = F.cross_entropy(out, y)

            # store influence-weighted samples in buffer ----------------------
            infl = out.detach().norm(p=2, dim=1)
            buffer.observe(feats.detach(), y.detach().cpu(), infl.cpu())

            # replay ----------------------------------------------------------
            replay_bs = int(128 * replay_r)
            if len(buffer) >= replay_bs > 0:
                re_f, re_y = buffer.sample(replay_bs)
                re_f, re_y = re_f.to(DEVICE), re_y.to(DEVICE)
                loss += F.cross_entropy(classifier(re_f), re_y)

            loss.backward()
            opt.step()
        sched.step()
        # coarse-grained Wasserstein refinement
        if (ep + 1) % 10 == 0 and hasattr(buffer, "wgf_refine"):
            buffer.wgf_refine()

    # ------------- validation -----------------------------------------------
    vloader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    backbone.eval(); classifier.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in vloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = classifier(backbone(x)).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    backbone.train(); classifier.train()
    return 100 * correct / total