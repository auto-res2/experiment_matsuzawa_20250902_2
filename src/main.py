"""src/main.py
Orchestrates the experiments. This file mirrors the behaviour of the original
single-file script while relying on the refactored helper modules.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .preprocess import MemThroughputProfiler, build_imagenet_loader, set_seed
from .train import get_model, train_epoch
from .evaluate import evaluate_cls, save_barplot, save_lineplot

# Optional heavy dependencies are imported lazily to avoid unnecessary memory
# usage – identical to the behaviour in the original script.
try:
    import detectron2  # noqa: F401 – only presence check is needed
except ImportError:
    detectron2 = None

# ---------------------------------------------------------------------------
#  EXPERIMENT 1 – Peak Memory & Resolution Scaling
# ---------------------------------------------------------------------------

def experiment1():
    print("\n========== Experiment 1: Peak-Memory & Resolution Scaling ==========")
    print(
        "Goal: measure peak GPU memory and throughput of VMamba-B vs. S²-Mamba-B "
        "for resolutions 224²…4096² on Tesla-T4."
    )

    resolutions = [224, 512, 1024, 2048, 4096]
    batch_schedule = {224: 8, 512: 8, 1024: 1, 2048: 1, 4096: 1}
    tile_schedule = {224: 64, 512: 64, 1024: 64, 2048: 128, 4096: 256}

    mem_vm, mem_s2, ips_vm, ips_s2 = [], [], [], []

    for res in resolutions:
        for model_name in ["VMambaB", "S2MambaB"]:
            set_seed(42)
            model = get_model(model_name)
            model = model.cuda() if torch.cuda.is_available() else model
            model.train()
            if model_name == "S2MambaB" and hasattr(model, "set_chunk"):
                model.set_chunk(tile_schedule[res])

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05
            )
            scaler = GradScaler()

            loader = build_imagenet_loader(
                res, batch_schedule[res], synthetic=res >= 2048
            )
            profiler = MemThroughputProfiler()
            profiler.reset()

            for it, (x, y) in enumerate(loader):
                if it >= 100:
                    break
                x = x.cuda(non_blocking=True).float() if torch.cuda.is_available() else x
                y = y.cuda(non_blocking=True) if torch.cuda.is_available() else y
                profiler.update(x.size(0))

                optimizer.zero_grad(set_to_none=True)
                with autocast(dtype=torch.float16):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            summary = profiler.summary()
            print(
                f"{model_name} @ {res}²  →  Peak mem: {summary['peak_mem_MiB']} MiB | "
                f"Throughput: {summary['img_per_sec']} img/s"
            )

            if model_name == "VMambaB":
                mem_vm.append(summary["peak_mem_MiB"])
                ips_vm.append(summary["img_per_sec"])
            else:
                mem_s2.append(summary["peak_mem_MiB"])
                ips_s2.append(summary["img_per_sec"])

            del model
            torch.cuda.empty_cache()

    save_lineplot(
        resolutions,
        {"VMamba-B": mem_vm, "S²-Mamba-B": mem_s2},
        title="Peak GPU memory vs. Resolution",
        xlabel="Resolution (pixels)",
        ylabel="Peak memory (MiB)",
        fname="peak_memory_scaling.pdf",
    )
    save_lineplot(
        resolutions,
        {"VMamba-B": ips_vm, "S²-Mamba-B": ips_s2},
        title="Throughput vs. Resolution",
        xlabel="Resolution (pixels)",
        ylabel="Images / second",
        fname="throughput_scaling.pdf",
    )

    print("Figures saved: peak_memory_scaling.pdf, throughput_scaling.pdf")
    print("===============================================================\n")


# ---------------------------------------------------------------------------
#  EXPERIMENT 2A – ImageNet Accuracy Retention
# ---------------------------------------------------------------------------

def experiment2a():
    print("\n========== Experiment 2a: ImageNet Accuracy Retention ==========")
    print(
        "Fine-tune VMamba-B vs. S²-Mamba-B on ImageNet-1K at 384² for 1 epoch "
        "(demo) to showcase identical accuracy under streaming execution."
    )

    res, batch = 384, 32
    loader_train = build_imagenet_loader(res, batch, split="train", synthetic=False)
    loader_val = build_imagenet_loader(res, batch, split="val", synthetic=False)

    results: Dict[str, Dict] = {}
    for model_name in ["VMambaB", "S2MambaB"]:
        set_seed(42)
        model = get_model(model_name)
        if model_name == "S2MambaB" and hasattr(model, "set_chunk"):
            model.set_chunk(64)
        model = model.cuda() if torch.cuda.is_available() else model

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
        scaler = GradScaler()

        train_stats, avg_loss = train_epoch(model, loader_train, optimizer, scaler)
        top1 = evaluate_cls(model, loader_val)
        results[model_name] = {
            "top1": top1,
            **train_stats,
            "loss": avg_loss,
        }
        print(
            f"{model_name}: Top-1={top1:.2f} %, PeakMem={train_stats['peak_mem_MiB']} MiB"
        )

        del model
        torch.cuda.empty_cache()

    save_barplot(
        list(results.keys()),
        [v["top1"] for v in results.values()],
        title="ImageNet-384 Top-1 Accuracy",
        ylabel="Top-1 (%)",
        fname="imagenet_accuracy.pdf",
    )
    print("Figure saved: imagenet_accuracy.pdf")
    print("===============================================================\n")


# ---------------------------------------------------------------------------
#  EXPERIMENT 2B – Detectron2 COCO demo (stub)
# ---------------------------------------------------------------------------

def experiment2b():
    if detectron2 is None:
        print("Detectron2 not found – skipping Experiment 2b (COCO).")
        return
    print(
        "Experiment 2b would run Detectron2 Mask R-CNN training – implementation "
        "omitted for brevity in this demo refactor."
    )


# ---------------------------------------------------------------------------
#  EXPERIMENT 3 – Ablation Study (simplified)
# ---------------------------------------------------------------------------

def experiment3():
    print("\n========== Experiment 3: Component Ablation Study ==========")
    variants = {
        "A_full": dict(cache=True, reversible=True, padding=True),
        "B_noCache": dict(cache=False, reversible=True, padding=True),
        "C_checkpoint": dict(cache=True, reversible=False, padding=True),
        "D_noPad": dict(cache=True, reversible=True, padding=False),
    }
    res, batch = 1024, 1
    loader = build_imagenet_loader(res, batch, split="val", synthetic=True)

    mems, times = [], []
    for vname, cfg in variants.items():
        print(f"Running variant {vname} …")
        set_seed(42)
        model = get_model("S2MambaB")
        if hasattr(model, "configure_stream"):
            model.configure_stream(
                chunk=64,
                cache_state=cfg["cache"],
                rev_residual=cfg["reversible"],
                pad_rf=cfg["padding"],
            )
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()

        profiler = MemThroughputProfiler()
        profiler.reset()
        start = time.perf_counter()
        with torch.no_grad(), autocast(dtype=torch.float16):
            for x, _ in loader:
                x = x.cuda(non_blocking=True).float() if torch.cuda.is_available() else x
                model(x)
                profiler.update(1)
                break  # single fwd pass suffices for memory measurement
        fwd_ms = (time.perf_counter() - start) * 1000
        stats = profiler.summary()

        mems.append(stats["peak_mem_MiB"])
        times.append(fwd_ms)
        print(f"{vname}: peakMem={stats['peak_mem_MiB']} MiB | fwdTime={fwd_ms:.1f} ms")

        del model
        torch.cuda.empty_cache()

    save_barplot(variants.keys(), mems, "Ablation – Peak Memory", "Peak Mem (MiB)", "ablation_memory.pdf")
    save_barplot(variants.keys(), times, "Ablation – Forward Time", "Time (ms)", "ablation_time.pdf")
    print("Figures saved: ablation_memory.pdf, ablation_time.pdf")
    print("===============================================================\n")


# ---------------------------------------------------------------------------
#  CLI PARSING & ENTRY POINT
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser("S²-Mamba Experiments (Tesla-T4)")
    p.add_argument("--experiment", choices=["exp1", "exp2a", "exp2b", "exp3"], required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    set_seed(args.seed)

    if args.experiment == "exp1":
        experiment1()
    elif args.experiment == "exp2a":
        experiment2a()
    elif args.experiment == "exp2b":
        experiment2b()
    elif args.experiment == "exp3":
        experiment3()
    else:
        raise ValueError("Unknown experiment")


if __name__ == "__main__":
    main()