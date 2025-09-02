"""src/main.py
Entry-point (`python -m src.main`). Orchestrates experiments while delegating
specialised work to train.py / evaluate.py / preprocess.py.
"""
from __future__ import annotations
import os, sys, time
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid", font_scale=1.1)  # noqa: E702
import torch

from src.train import (
    SEEDS,
    BF16_ENABLED,
    set_seed,
    build_model,
    train_one_epoch,
    peak_ram_gb,
)
from src.evaluate import validate, bar_plot, ci95
from src.preprocess import build_loader, DATA_ROOTS

# -----------------------------------------------------------------------------
# 0.  EARLY RESOURCE CHECK -----------------------------------------------------
# -----------------------------------------------------------------------------
CHECKPOINT_FILE = (
    Path := __import__("pathlib").Path  # inline import to keep header short
).home() / ".cache" / "timm" / "vmamba_tiny_patch16_224_ra3_in1k.pth"

missing_ds = [k for k, p in DATA_ROOTS.items() if not p.exists()]
if missing_ds:
    print("Required dataset directories not found:", ", ".join(missing_ds))
    sys.exit(31)
if not CHECKPOINT_FILE.exists():
    print("VMamba checkpoint missing – ensure timm>=0.9.16 cache is populated")
    sys.exit(31)

# CI smoke-test switch – NEVER used in prod runs --------------------------------
DEV_RUN = os.getenv("DEV_RUN", "0") == "1"

# -----------------------------------------------------------------------------
# 1.  EXPERIMENT 1  –  Quality / Efficiency parity -----------------------------
# -----------------------------------------------------------------------------

def exp1() -> None:
    print("\n" + "=" * 90)
    print("Experiment 1 – ImageNet-1K QUALITY / EFFICIENCY PARITY")
    print("=" * 90)

    device = torch.device("cuda")
    EPOCHS = 2 if DEV_RUN else 300

    records = []
    for impl in ("baseline", "flash"):
        for seed in SEEDS:
            set_seed(seed)
            model = build_model(impl, chunk=256).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
            scaler = torch.cuda.amp.GradScaler(enabled=BF16_ENABLED)

            train_loader, _ = build_loader("imagenet", "train", 128)
            val_loader, _ = build_loader("imagenet", "val", 256)

            thr_hist, mem_hist, top1_hist = [], [], []
            t_start = time.time()
            for ep in range(EPOCHS):
                mem = train_one_epoch(model, train_loader, opt, scaler, ep, EPOCHS)
                top1, _ = validate(model, val_loader)
                mem_hist.append(mem)
                top1_hist.append(top1)
                imgs_s = (128 * len(train_loader)) / (time.time() - t_start)
                thr_hist.append(imgs_s)

            records.append(
                {
                    "impl": impl,
                    "seed": seed,
                    "peak": float(np.mean(mem_hist)),
                    "ips": float(np.median(thr_hist[-200:])),
                    "top1": float(max(top1_hist)),
                }
            )
            print(
                f"{impl.title()}  seed={seed}  Top-1={records[-1]['top1']:.2f}  RAM={records[-1]['peak']:.2f} GB  img/s={records[-1]['ips']:.1f}"
            )

    df = pd.DataFrame(records)
    base, flash = df[df.impl == "baseline"], df[df.impl == "flash"]

    # paired stats -----------------------------------------------------------
    p_norm_peak = min(stats.shapiro(base.peak).pvalue, stats.shapiro(flash.peak).pvalue)
    mem_p = (
        stats.ttest_rel(base.peak, flash.peak).pvalue if p_norm_peak > 0.05 else stats.wilcoxon(base.peak, flash.peak).pvalue
    )
    ips_p = stats.ttest_rel(base.ips, flash.ips).pvalue
    acc_p = stats.ttest_rel(base.top1, flash.top1).pvalue

    def _line(metric: str, label: str, p: float) -> None:
        a, b = ci95(base[metric]); c, d = ci95(flash[metric])
        print(f"{label:<12s} Baseline {a:.2f}±{b:.2f}   Flash {c:.2f}±{d:.2f}   p={p:.3e}")

    print("\nExperimental numerical data:")
    _line("peak", "Peak-RAM", mem_p)
    _line("ips", "Images/s", ips_p)
    _line("top1", "Top-1", acc_p)

    # Figures ---------------------------------------------------------------
    bar_plot("memory_peak.pdf", {"baseline": base.peak.mean(), "flash": flash.peak.mean()}, "Peak GPU Memory", "GB")
    bar_plot("throughput_exp1.pdf", {"baseline": base.ips.mean(), "flash": flash.ips.mean()}, "Training Throughput", "img/s")
    bar_plot("accuracy_flash_vs_base.pdf", {"baseline": base.top1.mean(), "flash": flash.top1.mean()}, "Top-1 Accuracy", "%")

    print("Figures: memory_peak.pdf, throughput_exp1.pdf, accuracy_flash_vs_base.pdf")

# -----------------------------------------------------------------------------
# 2.  EXPERIMENT 2 – Scaling / OOM on 16 GB ------------------------------------
# -----------------------------------------------------------------------------

def _try_fw_bw(model_fn: Callable[[], torch.nn.Module], batch: int) -> Tuple[bool, float]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = model_fn().to(device)
    try:
        x = torch.randn(batch, 3, 224, 224, device=device)
        y = model(x)
        y.mean().backward()
        peak = peak_ram_gb()
        return True, peak
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, peak_ram_gb()
        raise


def exp2() -> None:
    print("\n" + "=" * 90)
    print("Experiment 2 – \"FIT vs OOM\" Scaling Study (16 GB)")
    print("=" * 90)

    configs = {
        "Baseline-TinyPlus32": lambda: build_model("baseline"),
        "Flash-TinyPlus32": lambda: build_model("flash", 256),
        "Baseline-Small32": lambda: build_model("baseline"),
        "Flash-Small32": lambda: build_model("flash", 256),
    }

    results = {}
    for name, fn in configs.items():
        lo, hi = 8, 256
        fit = False
        best = lo
        peak = float("nan")
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, mem = _try_fw_bw(fn, mid)
            if ok:
                fit, best, peak = True, mid, mem
                lo = mid + 1
            else:
                hi = mid - 1
        results[name] = {"fit": fit, "batch": best, "peak": peak}
        status = "OK" if fit else "OOM"
        print(f"{name:<22s} {status}  max-batch {best:3d}  peak {peak:.2f} GB")

    # Plot peaks for the configurations that *fit*
    mem_plot = {k: v["peak"] for k, v in results.items() if v["fit"]}
    if mem_plot:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(mem_plot.keys(), mem_plot.values())
        for b, v in zip(bars, mem_plot.values()):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("GB")
        ax.set_title("Peak Memory (Fittable models)")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig("scaling_memory.pdf", format="pdf", bbox_inches="tight")
        plt.close()
        print("Figure: scaling_memory.pdf")

# -----------------------------------------------------------------------------
# 3.  EXPERIMENT 3 – 3-way ANOVA Ablation --------------------------------------
# -----------------------------------------------------------------------------

def exp3() -> None:
    print("\n" + "=" * 90)
    print("Experiment 3 – Robustness & Ablation (3-way ANOVA)")
    print("=" * 90)
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    device = torch.device("cuda")
    EPOCHS = 1 if DEV_RUN else 90

    revs = [0, 1]
    chunks = [64, 128, 256, 512]
    gates = ["none", "0.5"]

    rows = []
    for r in revs:
        for c in chunks:
            for g in gates:
                seed = 0
                set_seed(seed)
                model = build_model("flash" if r else "baseline", chunk=c).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
                scaler = torch.cuda.amp.GradScaler(enabled=BF16_ENABLED)
                train_loader, _ = build_loader("imagenet100", "train", 64)
                val_loader, _ = build_loader("imagenet100", "val", 128)
                mem = train_one_epoch(model, train_loader, opt, scaler, 0, EPOCHS)
                top1, _ = validate(model, val_loader)
                rows.append({"rev": r, "chunk": c, "gate": g, "mem": mem, "top1": top1})
                print(f"rev={r} chunk={c} gate={g}  mem={mem:.2f} GB  Top-1={top1:.2f}")

    df = pd.DataFrame(rows)

    aov_mem = smf.ols("mem ~ C(rev)*C(chunk)*C(gate)", data=df).fit()
    aov_top = smf.ols("top1 ~ C(rev)*C(chunk)*C(gate)", data=df).fit()
    print("\nANOVA – Peak-RAM\n", anova_lm(aov_mem, typ=2))
    print("\nANOVA – Top-1\n", anova_lm(aov_top, typ=2))

    # Scatter memory vs chunk ------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.scatterplot(df, x="chunk", y="mem", hue="rev", style="gate", ax=ax)
    for _, r in df.iterrows():
        ax.text(r.chunk + 3, r.mem + 0.02, f"{r.top1:.1f}", fontsize=7)
    ax.set_title("Memory vs Chunk length (Exp-3)")
    ax.set_ylabel("GB")
    plt.tight_layout()
    plt.savefig("ablation_memory_chunk.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Figure: ablation_memory_chunk.pdf")

# -----------------------------------------------------------------------------
# 4.  MAIN ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        print("CUDA device not found – GPU experiments cannot proceed.")
        sys.exit(32)

    exp1()
    exp2()
    exp3()
    print("\nAll experiments finished – see printed tables & PDF figures for results.")


if __name__ == "__main__":
    main()
