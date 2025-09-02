"""
evaluate.py
All evaluation utilities + three experiments.
"""
from __future__ import annotations
import time, random
from pathlib import Path
from typing import Tuple, Dict, List

try:
    import torch
    import numpy as np, pandas as pd, scipy.stats as st
    import seaborn as sns, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ptflops
    from torchmetrics.functional import accuracy as tm_accuracy
except Exception as e:
    raise RuntimeError("Missing required Python libraries – aborting: " + str(e))

from .train import DEVICE, VARIANTS, load_model, LoFTAdapter
from .preprocess import build_loader, data_root

SEEDS = [11, 17, 23]
fig_root = Path("figures"); fig_root.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# Metric helpers -------------------------------------------------------------
clean_baseline_cache: Dict[str,float] = {}

def accuracy(logits, targets):
    return tm_accuracy(logits.softmax(dim=1), targets, task="multiclass", num_classes=1000)

def compute_mce(err_rate: float, corruption: str):
    if corruption not in clean_baseline_cache:
        raise KeyError(f"Baseline error for corruption '{corruption}' not computed")
    return 100. * err_rate / clean_baseline_cache[corruption]

# ---------------------------------------------------------------------------
# Low-level evaluation utility ------------------------------------------------

def evaluate_acc(model: torch.nn.Module, loader):
    acc_meter = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            acc_meter.append(accuracy(logits, y).cpu())
    acc = torch.mean(torch.stack(acc_meter)).item()
    return acc, 1-acc

# ---------------------------------------------------------------------------
# EXPERIMENT 1 – Real-data robustness benchmark ------------------------------

def exp1_run():
    print("\n================  EXPERIMENT 1 – REAL DATA BENCHMARK  ================" )
    arch = "resnet50"
    # 0) baseline error per corruption ------------------------------------
    print("Computing baseline errors …", flush=True)
    base_model = load_model(arch, "vanilla", seed=11)
    _ = evaluate_acc(base_model, build_loader("clean", 64))[0]
    for corr in ["gaussian_noise", "defocus_blur", "jpeg", "brightness", "snow", "speckle_noise", "glass_blur", "fog"]:
        err = 1 - evaluate_acc(base_model, build_loader(f"C/{corr}/s3", 64))[0]
        clean_baseline_cache[corr] = err
    # 1) evaluate all variants & seeds ------------------------------------
    records = []
    for variant in VARIANTS:
        for seed in SEEDS:
            mdl = load_model(arch, variant, seed)
            clean_acc, _ = evaluate_acc(mdl, build_loader("clean",64))
            es_acc, _    = evaluate_acc(mdl, build_loader("ES",64))
            mce_vals = []
            for corr in clean_baseline_cache:
                acc, _ = evaluate_acc(mdl, build_loader(f"C/{corr}/s3",64))
                mce_vals.append(compute_mce(1-acc, corr))
            mce_mean = np.mean(mce_vals)
            records.append(dict(variant=variant, seed=seed, clean=clean_acc*100,
                                es=es_acc*100, mce=mce_mean))
    df = pd.DataFrame(records)
    print("\nRaw scores (mean±sd over seeds):")
    # --- FIXED: Correct bracket placement ---------------------------------
    stats_df = df.groupby("variant").agg(["mean","std"]).round(2)
    print(stats_df[[("clean","mean"),("mce","mean"),("es","mean")]])
    # paired t-test ---------------------------------------------------------
    best_base = df[df.variant=="augmix"].set_index("seed")["mce"]
    loft_vals = df[df.variant=="loft"].set_index("seed")["mce"]
    t,p = st.ttest_rel(best_base, loft_vals)
    print(f"\nPaired t-test LoFT vs AugMix:  t={t:.2f}  p={p:.4f}")
    # Figure ---------------------------------------------------------------
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6,4))
    bar = df.groupby("variant")["mce"].mean().loc[VARIANTS]
    ax.bar(range(len(bar)), bar)
    ax.set_xticks(range(len(bar)))
    ax.set_xticklabels(VARIANTS, rotation=20)
    ax.set_ylabel("mCE (↓)")
    ax.set_title("Mean Corruption Error – ResNet-50")
    for i,v in enumerate(bar):
        ax.text(i, v+0.4, f"{v:.1f}", ha='center')
    fname = fig_root/"mCE_resnet_loft_vs_baselines.pdf"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Figure saved: {fname}")
    print("====================================================================\n")

# ---------------------------------------------------------------------------
# EXPERIMENT 2 – Ablations ---------------------------------------------------

def make_ablation(tag: str):
    base = __import__("torchvision").models.resnet50(weights=None)
    if tag == "full":
        return LoFTAdapter(base)
    elif tag == "minus_freq":
        mdl = LoFTAdapter(base)
        mdl.enc.proj.weight.data[:12].zero_()
        return mdl
    elif tag == "fixed_gb":
        return base
    else:
        raise ValueError(tag)

def exp2_run():
    print("\n================  EXPERIMENT 2 – CAUSAL ABLATIONS  ==================")
    variants = {k: make_ablation(k) for k in ["full","minus_freq","fixed_gb"]}
    severities = [1,2,3,4,5]
    corruption = "defocus_blur"
    curves: Dict[str,List[float]] = {k:[] for k in variants}
    for sev in severities:
        split = f"C/{corruption}/s{sev}"
        loader = build_loader(split, 64)
        for tag, mdl in variants.items():
            mdl.to(DEVICE).eval()
            acc,_ = evaluate_acc(mdl, loader)
            curves[tag].append(acc*100)
    df = pd.DataFrame(curves, index=[f"sev{v}" for v in severities])
    print(df.round(2))
    fig, ax = plt.subplots(figsize=(6,4))
    for tag, vals in curves.items():
        ax.plot(severities, vals, marker='o', label=tag)
        for s,v in zip(severities, vals):
            ax.text(s, v+0.3, f"{v:.1f}", ha='center', fontsize=7)
    ax.set_xlabel("Severity"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Severity – Defocus Blur")
    ax.legend()
    fname = fig_root/"mCE_severity_blur.pdf"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Figure saved: {fname}")
    print("====================================================================\n")

# ---------------------------------------------------------------------------
# EXPERIMENT 3 – Efficiency profiling ---------------------------------------

def _profile(model: torch.nn.Module, bs: int) -> Tuple[float,float]:
    dummy = torch.randn(bs,3,224,224, device=DEVICE)
    model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    reps = 100
    starter.record()
    for _ in range(reps):
        model(dummy)
    ender.record();
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    lat = starter.elapsed_time(ender)/reps
    flops, _ = ptflops.get_model_complexity_info(model.cpu(), (3,224,224), as_strings=False, print_per_layer_stat=False)
    model.to(DEVICE)
    return lat, flops/1e9

def exp3_run():
    print("\n================  EXPERIMENT 3 – EFFICIENCY PROFILING  ===============")
    models = {
        "vanilla": __import__("torchvision").models.resnet50(weights=None),
        "loft"   : LoFTAdapter(__import__("torchvision").models.resnet50(weights=None))
    }
    res = {}
    for tag, mdl in models.items():
        mdl.eval().to(DEVICE)
        lat1, fl1 = _profile(mdl, 1)
        lat64, _  = _profile(mdl, 64)
        params = sum(p.numel() for p in mdl.parameters())/1e6
        res[tag] = dict(lat1=round(lat1,2), lat64=round(lat64,2), GFLOPs=round(fl1,2), params=round(params,2))
    df = pd.DataFrame(res).T
    print(df)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(df.index, df.lat1)
    for i,v in enumerate(df.lat1):
        ax.text(i, v+0.1, f"{v:.2f}", ha='center')
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Batch-1 Inference Latency")
    fname = fig_root/"inference_latency.pdf"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Figure saved: {fname}")
    print("====================================================================\n")
