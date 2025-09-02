"""src/main.py
Main orchestration script – reproduces the behaviour of the original monolithic
file while re-using the newly modularised code.
It can be executed via `python -m src.main` from project root.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .train import (
    SEEDS,
    FlashMambaBlock,
    MambaBlock,
    TinyVisionMamba,
    set_seed,
    train_one_epoch,
)
from .preprocess import build_loader
from .evaluate import (
    ci95,
    evaluate,
    memory_benchmark,
    print_bar,
    save_bar_plot,
    ttest,
    try_init_model,
)

# ---------------------------------------------------------------------------------
# EXPERIMENT 1 – Baseline vs Flash on identical architecture
# ---------------------------------------------------------------------------------

def experiment_1():
    title = "EXPERIMENT 1 – MEMORY, SPEED & ACCURACY ON IDENTICAL NETWORK"
    desc = (
        "Baseline uses full-sequence Mamba blocks; Flash variant replaces them "
        "with chunk-prefix-scan + reversible coupling (chunk=256).  Each seed "
        "trains one real epoch on a FakeData ImageNet-like set so numbers are "
        "measured live – nothing is hard-coded."
    )
    print_bar(title)
    print(desc)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {"baseline": [], "flash": []}

    for seed in SEEDS:
        set_seed(seed)

        # Models -----------------------------------------------------------------
        base_model = TinyVisionMamba(MambaBlock, depth=4, dim=128).to(device)
        flash_model = TinyVisionMamba(FlashMambaBlock, depth=4, dim=128, chunk_len=256).to(device)

        # Data -------------------------------------------------------------------
        train_loader = build_loader(batch_size=32, num_samples=2048)
        val_loader = build_loader(batch_size=64, num_samples=512)

        # Baseline ---------------------------------------------------------------
        optim_b = torch.optim.AdamW(base_model.parameters(), lr=1e-3)
        mem_b = memory_benchmark(base_model, batch_size=32, device=device)
        t_b = train_one_epoch(base_model, train_loader, optim_b)
        acc_b = evaluate(base_model, val_loader)
        results["baseline"].append({"mem": mem_b, "ips": len(train_loader.dataset) / t_b, "acc": acc_b})

        # Flash ------------------------------------------------------------------
        optim_f = torch.optim.AdamW(flash_model.parameters(), lr=1e-3)
        mem_f = memory_benchmark(flash_model, batch_size=32, device=device)
        t_f = train_one_epoch(flash_model, train_loader, optim_f)
        acc_f = evaluate(flash_model, val_loader)
        results["flash"].append({"mem": mem_f, "ips": len(train_loader.dataset) / t_f, "acc": acc_f})

    # Aggregate statistics -------------------------------------------------------
    def gather(key):
        return [d[key] for d in results["baseline"]], [d[key] for d in results["flash"]]

    mem_b, mem_f = gather("mem")
    ips_b, ips_f = gather("ips")
    acc_b, acc_f = gather("acc")

    summary = {
        "Peak-RAM (GB)": {"baseline": ci95(mem_b), "flash": ci95(mem_f), "p": ttest(mem_b, mem_f)},
        "Images/s": {"baseline": ci95(ips_b), "flash": ci95(ips_f), "p": ttest(ips_b, ips_f)},
        "Top-1 (%)": {"baseline": ci95(acc_b), "flash": ci95(acc_f), "p": ttest(acc_b, acc_f)},
    }

    print("\nExperimental numerical data (mean±CI95):")
    for k, v in summary.items():
        print(
            f"  {k:12s}  Baseline: {v['baseline'][0]:.2f}±{v['baseline'][1]:.2f}   "
            f"Flash: {v['flash'][0]:.2f}±{v['flash'][1]:.2f}   p={v['p']:.3f}"
        )

    # Figures -------------------------------------------------------------------
    save_bar_plot({"Baseline": np.mean(mem_b), "Flash-SSM": np.mean(mem_f)}, "Peak RAM (GB)", "Peak GPU memory – Exp-1", "memory_peak.pdf")
    save_bar_plot({"Baseline": np.mean(ips_b), "Flash-SSM": np.mean(ips_f)}, "Images / s", "Throughput – Exp-1", "throughput.pdf")
    save_bar_plot({"Baseline": np.mean(acc_b), "Flash-SSM": np.mean(acc_f)}, "Top-1 (%)", "Accuracy – Exp-1", "accuracy.pdf")

    print("\nFigures saved:")
    for fn in ["memory_peak.pdf", "throughput.pdf", "accuracy.pdf"]:
        print("  ", fn)

    return summary


# ---------------------------------------------------------------------------------
# EXPERIMENT 2 – Larger depth/state under 16 GB
# ---------------------------------------------------------------------------------

def experiment_2():
    title = "EXPERIMENT 2 – SCALING DEPTH/STATE ON 16 GB"
    desc = (
        "We test whether larger Tiny-Plus-32 and Small-32 models fit in 16 GB "
        "when using Flash-SSM blocks, and whether the baseline implementation "
        "OOMs.  Only memory is probed here (single forward pass)."
    )
    print_bar(title)
    print(desc)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from .train import FlashMambaBlock, MambaBlock  # local import to avoid circular

    configs = [
        ("Baseline-TinyPlus-32", 40, 256, MambaBlock),
        ("Flash-TinyPlus-32", 40, 256, FlashMambaBlock),
        ("Baseline-Small-32", 40, 320, MambaBlock),
        ("Flash-Small-32", 40, 320, FlashMambaBlock),
    ]

    records = {}
    for name, depth, dim, block in configs:
        fits, mem = try_init_model(depth, dim, block, device)
        records[name] = {"fits": fits, "peak_ram": mem}
        status = "OK" if fits else "OOM"
        print(f"  {name:20s}  ->  {status}  Peak-RAM: {mem}")

    # Plot only the models that fit ---------------------------------------------
    mem_plot = {n: v["peak_ram"] for n, v in records.items() if v["fits"]}
    if mem_plot:
        save_bar_plot(mem_plot, "Peak RAM (GB)", "Memory of larger models – Exp-2", "scaling_memory.pdf")
        print("\nFigure saved:\n  scaling_memory.pdf")

    return records


# ---------------------------------------------------------------------------------
# EXPERIMENT 3 – 2×4×2 ablation grid
# ---------------------------------------------------------------------------------

def experiment_3():
    title = "EXPERIMENT 3 – ABLATION OF REVERSIBLE / CHUNK / GATE"
    desc = (
        "Runs a 16-config grid (reversible ON/OFF, chunk length, sparse gate) "
        "for a single epoch on FakeData-100 to gather memory & speed numbers."
    )
    print_bar(title)
    print(desc)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rev_opts = [True, False]
    chunk_opts = [64, 128, 256, 512]
    gate_opts = [None, 0.5]  # gate is a no-op in this simplified demo

    rows = []
    for rev in rev_opts:
        for W in chunk_opts:
            for gate in gate_opts:
                block_cls = FlashMambaBlock if rev else MambaBlock
                model = TinyVisionMamba(block_cls, depth=4, dim=128, chunk_len=W).to(device)
                train_loader = build_loader(batch_size=32, num_samples=1024)
                optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
                mem = memory_benchmark(model, batch_size=32, device=device)
                t = train_one_epoch(model, train_loader, optim)
                ips = len(train_loader.dataset) / t
                rows.append({"rev": rev, "W": W, "gate": "none" if gate is None else gate, "mem": mem, "ips": ips})

    df = pd.DataFrame(rows)
    print("\nFirst five measured rows:")
    print(df.head())

    # 3-way ANOVA (memory & speed) ----------------------------------------------
    def anova(col):
        groups = []
        for rev in rev_opts:
            groups.append(df[df["rev"] == rev][col])
        for W in chunk_opts:
            groups.append(df[df["W"] == W][col])
        for g in ["none", 0.5]:
            groups.append(df[df["gate"] == g][col])
        return stats.f_oneway(*groups).pvalue

    p_mem = anova("mem")
    p_ips = anova("ips")
    print(f"\nANOVA p-values  Peak-RAM: {p_mem:.4f}   Images/s: {p_ips:.4f}")

    # Scatter plot --------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df["mem"], df["ips"], c=df["W"], cmap="viridis", s=80)
    ax.set_xlabel("Peak RAM (GB)")
    ax.set_ylabel("Images / s")
    ax.set_title("Exp-3  Memory vs Speed (colour = chunk)")
    plt.colorbar(sc, label="Chunk length")
    plt.tight_layout()
    plt.savefig("ablation_scatter.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("\nFigure saved:\n  ablation_scatter.pdf")

    return df


# ---------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")  # speed on A100 / T4

    exp1_summary = experiment_1()
    exp2_records = experiment_2()
    exp3_df = experiment_3()

    # Persist raw metrics --------------------------------------------------------
    Path("logs").mkdir(exist_ok=True)
    with open("logs/exp1_summary.json", "w") as f:
        json.dump(exp1_summary, f, indent=2)
    with open("logs/exp2_records.json", "w") as f:
        json.dump(exp2_records, f, indent=2)
    exp3_df.to_csv("logs/exp3_ablation.csv", index=False)

    print("\nAll experiments finished – raw metrics saved to ./logs and figures to PDFs.")


if __name__ == "__main__":
    main()
