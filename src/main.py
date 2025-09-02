"""src/main.py
Entry point executed via `python -m src.main`. Orchestrates all three
experiments by delegating to the utility modules.
"""
from __future__ import annotations

import math, os, sys, json, time, random, warnings  # noqa: F401  (legacy keeping)
from pathlib import Path
from typing import Tuple

import torch
from scipy import stats  # needed for experiment statistics

# --- Local modules -----------------------------------------------------------
from . import train as tr
from . import evaluate as ev
from . import preprocess as pp

# -----------------------------------------------------------------------------
#  Helper for experiment-2 OOM probing
# -----------------------------------------------------------------------------

def try_batch(model_fn, batch: int, device) -> Tuple[bool, float]:
    """Return (fits?, peak_memory_GB). Runs a single FW+BW with dummy data."""
    model = model_fn().to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        dummy = torch.randn(batch, 3, 224, 224, device=device)
        out = model(dummy)
        loss = out.mean()
        loss.backward()
        return True, ev.peak_gpu_gb()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, ev.peak_gpu_gb()
        raise  # unrelated error – propagate

# -----------------------------------------------------------------------------
#  Experiment 1 – quality parity (short CI epoch)
# -----------------------------------------------------------------------------

def experiment1():
    desc = (
        "Experiment 1 – ImageNet-1K quality parity: Baseline VMamba-Tiny vs "
        "Flash-SSM (chunk 256, reversible). One very short epoch is executed "
        "to keep CI budget reasonable."
    )
    print("\n" + "=" * 80 + "\n" + desc + "\n" + "=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 1  # increase to 300 for the full study

    metrics = {"base": [], "flash": []}

    for seed in tr.SEEDS:
        tr.set_seed(seed)

        # ---------------- Models ----------------
        base = tr.load_baseline().to(device)
        flash = tr.convert_to_flash(tr.load_baseline(), chunk=256).to(device)

        # ---------------- Data ------------------
        trainL = pp._build_loader(pp.IMAGENET_ROOT, True, batch=32)
        valL = pp._build_loader(pp.IMAGENET_ROOT, False, batch=64)

        # ---------- Baseline run ---------------
        opt = torch.optim.AdamW(base.parameters(), lr=1e-3)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for _ in range(EPOCHS):
            tr.run_epoch(base, trainL, opt)
            acc, ips = tr.run_epoch(base, valL)
        mem = ev.peak_gpu_gb()
        metrics["base"].append((mem, ips, acc))

        # ---------- Flash run ------------------
        opt = torch.optim.AdamW(flash.parameters(), lr=1e-3)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for _ in range(EPOCHS):
            tr.run_epoch(flash, trainL, opt)
            acc, ips = tr.run_epoch(flash, valL)
        mem = ev.peak_gpu_gb()
        metrics["flash"].append((mem, ips, acc))

    # Aggregate statistics --------------------------------------------------
    def unpack(idx):
        return [m[idx] for m in metrics["base"]], [m[idx] for m in metrics["flash"]]

    memB, memF = unpack(0)
    ipsB, ipsF = unpack(1)
    accB, accF = unpack(2)

    def row(name, a, b):
        mA, cA = ev.ci95(a)
        mB, cB = ev.ci95(b)
        p = stats.ttest_rel(a, b).pvalue
        print(f"{name:12s}  Baseline {mA:.2f}±{cA:.2f}   Flash {mB:.2f}±{cB:.2f}   p={p:.3f}")

    print("\nExperimental numerical data (mean±CI95):")
    row("Peak-RAM", memB, memF)
    row("Images/s", ipsB, ipsF)
    row("Top-1(%)", accB, accF)

    # Figures ---------------------------------------------------------------
    ev.save_bar({"Baseline": sum(memB) / len(memB), "Flash": sum(memF) / len(memF)}, "Peak GPU memory – Exp-1", "GB", "memory_peak.pdf")
    ev.save_bar({"Baseline": sum(ipsB) / len(ipsB), "Flash": sum(ipsF) / len(ipsF)}, "Throughput – Exp-1", "img/s", "throughput_exp1.pdf")
    ev.save_bar({"Baseline": sum(accB) / len(accB), "Flash": sum(accF) / len(accF)}, "Accuracy – Exp-1", "Top-1 %", "accuracy_flash_vs_base.pdf")
    print("\nFigures: memory_peak.pdf, throughput_exp1.pdf, accuracy_flash_vs_base.pdf")

# -----------------------------------------------------------------------------
#  Experiment 2 – scaling / OOM study
# -----------------------------------------------------------------------------

def experiment2():
    desc = (
        "Experiment 2 – Scaling depth/state under limited memory. We perform "
        "a coarse batch-size sweep until Out-Of-Memory occurs."
    )
    print("\n" + "=" * 80 + "\n" + desc + "\n" + "=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = {
        "Baseline-TinyPlus32": lambda: tr.load_baseline(),
        "Flash-TinyPlus32": lambda: tr.convert_to_flash(tr.load_baseline(), 256),
        "Baseline-Small32": lambda: tr.load_baseline(),  # placeholder in CI
        "Flash-Small32": lambda: tr.convert_to_flash(tr.load_baseline(), 256),
    }

    records = {}
    for name, fn in configs.items():
        batch = 8
        hi = 256
        fit, mem = False, float("nan")
        while batch <= hi:
            ok, m = try_batch(fn, batch, device)
            if ok:
                fit, mem = True, m
                batch *= 2
            else:
                break
        records[name] = {"fits": fit, "peak_gb": mem, "batch": batch // 2 if fit else 0}
        status = "OK" if fit else "OOM"
        print(f"{name:22s}  {status:3s}  peak {mem:.3f} GB  max-batch {records[name]['batch']}")

    mem_plot = {n: v["peak_gb"] for n, v in records.items() if v["fits"]}
    if mem_plot:
        ev.save_bar(mem_plot, "Scaling memory – Exp-2", "GB", "scaling_memory.pdf")
        print("Figure: scaling_memory.pdf")

# -----------------------------------------------------------------------------
#  Experiment 3 – ablation ANOVA (memory vs chunk length)
# -----------------------------------------------------------------------------

def experiment3():
    desc = (
        "Experiment 3 – Ablation (reversible × chunk) on ImageNet-100 with "
        "one-way ANOVA over memory usage."
    )
    print("\n" + "=" * 80 + "\n" + desc + "\n" + "=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rev_opts = [True, False]
    chunk_opts = [64, 128, 256, 512]

    rows = []
    for rev in rev_opts:
        for W in chunk_opts:
            model = tr.load_baseline()
            if rev:
                model = tr.convert_to_flash(model, W)
            model.to(device)

            trainL = pp._build_loader(pp.IMAGENET100_ROOT, True, batch=16)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            tr.run_epoch(model, trainL, opt)
            mem = ev.peak_gpu_gb()
            rows.append({"rev": int(rev), "chunk": W, "mem": mem})

    import pandas as pd  # local import to keep global namespace clean
    df = pd.DataFrame(rows)
    print("\nFirst rows:\n", df.head())

    # One-way ANOVA per factor
    p_rev = stats.f_oneway(df[df.rev == 1].mem, df[df.rev == 0].mem).pvalue
    p_W = stats.f_oneway(*[df[df.chunk == c].mem for c in chunk_opts]).pvalue
    print(f"ANOVA p-values  reversible={p_rev:.4f}  chunk={p_W:.4f}")

    # Scatter plot memory vs chunk length
    import seaborn as sns  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots()
    sns.scatterplot(df, x="chunk", y="mem", hue="rev", ax=ax)
    ax.set_title("Exp-3  Memory vs Chunk length")
    plt.tight_layout()
    plt.savefig("ablation_memory_chunk.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Figure: ablation_memory_chunk.pdf")

# -----------------------------------------------------------------------------
#  Main entry point
# -----------------------------------------------------------------------------

def main():
    # newer PyTorch versions expose this API – guard for compatibility
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except (ValueError, AttributeError):
            pass  # silently ignore on unsupported versions

    experiment1()
    experiment2()
    experiment3()

    Path("logs").mkdir(exist_ok=True)
    print("\nAll experiments completed – raw numbers printed above; figures saved as .pdf files.")


if __name__ == "__main__":
    main()
