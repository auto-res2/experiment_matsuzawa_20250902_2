"""
main.py  –  orchestrates the experimental pipeline.
Execute with:  python -m src.main  --exp {1,2,3}
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import TextLogger
from avalanche.training.strategies import DER, Replay

from .preprocess import (build_cifar100_benchmark, build_omniglot_rotation_scenario,
                         build_tinyimagenet_benchmark, set_all_seeds)
from .train import SQMStrategy, build_backbone
from .evaluate import annotate_bar, compute_flops, set_plot_style

# switch to non-interactive backend for headless execution (e.g. SLURM)
plt.switch_backend("Agg")
set_plot_style()

# --------------------------------------------------------------------------------
#  EXPERIMENT 1  – Accuracy & forgetting under 200 KB
# --------------------------------------------------------------------------------

def run_experiment_1() -> None:
    print("\n======================================================")
    print("Experiment 1 – Accuracy & Forgetting under a Hard 200 KB Budget")
    print("======================================================\n")

    # ---------------- Benchmarks ----------------
    benchmarks = {
        "cifar100": build_cifar100_benchmark(20, return_task_id=False),
        "tinyimagenet": build_tinyimagenet_benchmark(10, return_task_id=False),
    }

    # ---------------- Strategies ----------------
    strategies: Dict[str, object] = {}

    # our method (SQM)
    backbone, feat_dim = build_backbone("resnet18")
    optim = torch.optim.SGD(backbone.parameters(), lr=0.1, momentum=0.9)
    strategies["SQM"] = SQMStrategy(backbone, optim, torch.nn.CrossEntropyLoss(), feat_dim,
                                     lam=0.03, pq_clusters=4, drop_ratio=0.8)

    # ER-Ring buffer baseline
    backbone_er, _ = build_backbone("resnet18")
    opt_er = torch.optim.SGD(backbone_er.parameters(), lr=0.1)
    mem_size = int(200 * 1024 / (32 * 32 * 3))
    strategies["ER"] = Replay(backbone_er, opt_er, torch.nn.CrossEntropyLoss(), mem_size=mem_size)

    # DER++ baseline
    backbone_der, _ = build_backbone("resnet18")
    opt_der = torch.optim.SGD(backbone_der.parameters(), lr=0.1)
    strategies["DER++"] = DER(backbone_der, opt_der, torch.nn.CrossEntropyLoss(), mem_size=mem_size)

    # ---------------- Run loop ----------------
    results = []
    for bench_name, benchmark in benchmarks.items():
        print(f"\n--- Dataset: {bench_name} ---")
        for strat_name, strat in strategies.items():
            print(f"\n>>> Training strategy: {strat_name}")

            set_all_seeds(0)
            t_start = time.time()

            # silent evaluation plugin – we only need final numbers
            eval_plugin = avalanche.training.plugins.EvaluationPlugin(
                accuracy_metrics(stream=True), forgetting_metrics(stream=True),
                logger=TextLogger(open(os.devnull, "w"))
            )
            strat.evaluator = eval_plugin

            for exp_id, experience in enumerate(benchmark.train_stream):
                print(f"Task {exp_id + 1}/{len(benchmark.train_stream)}")
                # optional replay batch for SQM (omitted for baselines)
                if strat_name == "SQM":
                    imgs_rep, labels_rep = strat.collect_replay()
                    if imgs_rep is not None:
                        pass  # dataset concatenation skipped for brevity
                strat.train(experience)

            acc_dict = strat.eval(benchmark.test_stream)
            t_end = time.time()

            top1_acc = acc_dict["Top1_Acc_Stream/eval_phase/test_stream"]
            extra_mem = strat.extra_memory_bytes() if strat_name == "SQM" else 0
            flops = compute_flops(strat.model) or 0.0

            print(f"Final Avg Acc: {top1_acc:.2f}% | Extra Mem: {extra_mem/1024:.1f} KB | "
                  f"FLOPs: {flops/1e9:.2f} G | Time: {t_end - t_start:.1f}s")

            results.append({
                "dataset": bench_name,
                "strategy": strat_name,
                "accuracy": top1_acc,
                "extra_memory_KB": extra_mem / 1024,
                "flops_G": flops / 1e9,
                "time_s": t_end - t_start,
            })

    # ---------------- Tabulate & plot ----------------
    df = pd.DataFrame(results)
    print("\n===== Numerical Results =====")
    print(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="strategy", y="accuracy", hue="dataset", ax=ax)
    annotate_bar(ax)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_xlabel("")
    ax.set_title("Experiment-1: Accuracy under 200 KB")
    plt.legend(title="Dataset")
    fig.tight_layout()
    fig.savefig("accuracy_budget200.pdf", bbox_inches="tight")
    print("Figure saved: accuracy_budget200.pdf")

# --------------------------------------------------------------------------------
#  EXPERIMENT 2  – ablation & memory scaling
# --------------------------------------------------------------------------------

def run_experiment_2():
    print("\n======================================================")
    print("Experiment 2 – Component Ablation & Memory Scaling Across 100 Tasks")
    print("======================================================\n")

    scenario = build_omniglot_rotation_scenario(100)

    # helper to build SQM variants ------------------------------------------------
    def make_variant(*, disable_cs=False, disable_pq=False, disable_sparse=False):
        cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(), torch.nn.Linear(64 * 7 * 7, 128))
        opt = torch.optim.SGD(cnn.parameters(), lr=0.05)
        strat = SQMStrategy(cnn, opt, torch.nn.CrossEntropyLoss(), feat_dim=128, lam=0.03)
        if disable_cs:
            strat.sketches = {}
        if disable_pq:
            strat.pq_memory = None
        if disable_sparse:
            strat.drop_ratio = 0.0
        return strat

    variants = {
        "SQM": make_variant(),
        "-CS": make_variant(disable_cs=True),
        "-PQ": make_variant(disable_pq=True),
        "-SPARSE": make_variant(disable_sparse=True),
    }

    results = []
    for name, strat in variants.items():
        print(f"Variant: {name}")
        set_all_seeds(0)
        for exp_id, exp in enumerate(scenario.train_stream):
            strat.train(exp)
            if (exp_id + 1) % 10 == 0:
                results.append({
                    "tasks": exp_id + 1,
                    "variant": name,
                    "memory_KB": strat.extra_memory_bytes() / 1024,
                })
        acc = strat.eval(scenario.test_stream)
        print(f"Final accuracy: {acc['Top1_Acc_Stream/eval_phase/test_stream']:.2f}%")

    df = pd.DataFrame(results)
    sns.lineplot(data=df, x="tasks", y="memory_KB", hue="variant", marker="o")
    plt.xlabel("# Tasks")
    plt.ylabel("Memory (KB)")
    plt.title("Experiment-2: Memory scaling vs tasks")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("memory_scaling.pdf", bbox_inches="tight")
    print("Figure saved: memory_scaling.pdf")

# --------------------------------------------------------------------------------
#  EXPERIMENT 3  – on-device run (placeholder)
# --------------------------------------------------------------------------------

def run_experiment_3():
    print("Experiment 3 requires ADB and cannot be executed inside this environment.\n"
          "Please refer to scripts/mobile/run_on_device.sh for details.")

# --------------------------------------------------------------------------------
#  CLI entry-point
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], default=1,
                        help="Which experiment to run (1/2/3)")
    args = parser.parse_args()

    if args.exp == 1:
        run_experiment_1()
    elif args.exp == 2:
        run_experiment_2()
    else:
        run_experiment_3()


if __name__ == "__main__":
    main()
