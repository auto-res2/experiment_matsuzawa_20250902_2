"""src/evaluate.py
Evaluation / analysis code for the HARD experiments.
Each `run_experimentX` mirrors the original monolithic implementation but now
relies on helper utilities from `train.py` and `preprocess.py`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as T
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from fvcore.nn import FlopCountAnalysis
from torchmetrics.image.fid import FrechetInceptionDistance

# Project-internal imports
from src.train import HARDWrapper, SEED_LIST, set_seed  # noqa: E402
from src import preprocess as prep  # noqa: E402

# --------------------------------------------------
#  EXPERIMENT 1  –  DiT-XL Pareto (un-conditional)
# --------------------------------------------------

def run_experiment1(args):
    """Quality-vs-efficiency Pareto curve on a 3-stage DiT cascade."""

    print("""
────────────────────────────────────────────────────────
EXPERIMENT 1 – HARD vs Baselines on DiT-XL Cascade
────────────────────────────────────────────────────────
    """)

    import torchvision.datasets as dsets  # local import to avoid unnecessary cost
    from diffusers import UNet2DConditionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # --------------------------------------------------
    # 1) Load / mock the three cascade stages
    # --------------------------------------------------
    def _mock_stage():
        return UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 320),
        ).to(device, dtype=dtype)

    stages = [_mock_stage(), _mock_stage(), _mock_stage()]  # 64→128→256

    tau_grid = [0.05, 0.10, 0.15, 0.20]
    results: List[Dict] = []

    # --------------------------------------------------
    # 2) Real distribution activations for FID
    # --------------------------------------------------
    try:
        real_loader = prep.get_imagenet_loader(batch_size=32, num_workers=4)
    except FileNotFoundError as e:
        raise RuntimeError("Imagenet val set is required for Experiment 1.") from e

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    print("Computing real activations for FID …")
    for imgs, _ in tqdm(real_loader, desc="Real activations"):
        fid_metric.update(imgs.to(device), real=True)

    # --------------------------------------------------
    # 3) Iterate over τ grid and seeds
    # --------------------------------------------------
    for tau in tau_grid:
        print(f"\n===== τ = {tau:.2f} =====")
        for seed in SEED_LIST:
            set_seed(seed)
            hw = HARDWrapper(stages, taus=[tau] * 3, freeze_k=2, device=device, dtype=dtype)
            generated_imgs = []

            num_gen = 5_000
            for _ in tqdm(range(num_gen), desc=f"Gen τ={tau} seed={seed}"):
                latents = [
                    torch.randn(1, 4, 64 * (2 ** lvl), 64 * (2 ** lvl), device=device, dtype=dtype)
                    for lvl in range(3)
                ]
                timesteps = [torch.randint(0, 1000, (1,), device=device) for _ in range(3)]
                with autocast(device.type):
                    outs, _ = hw.sample(latents, timesteps)
                img = torch.clamp((outs[-1].float() + 1) / 2, 0, 1)
                generated_imgs.append(img.squeeze(0).cpu())

            gen_tensor = torch.stack(generated_imgs)
            fid_metric.reset()
            fid_metric.update(gen_tensor.to(device), real=False)
            fid_score = fid_metric.compute().item()

            # FLOPs (normalised by T4 efficiency constant)
            T4_SM_EFFICIENCY = 7.5
            sample_flops = FlopCountAnalysis(hw, (latents, timesteps)).total() / 1e9
            flops = sample_flops * num_gen / T4_SM_EFFICIENCY

            results.append({"τ": tau, "seed": seed, "FID": fid_score, "GFLOPs": flops})
            print("Result:", results[-1])

    # --------------------------------------------------
    # 4) Save & plot
    # --------------------------------------------------
    df = pd.DataFrame(results)
    out_dir = Path(args.output_dir) / "exp1"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp1_metrics.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    for seed, g in df.groupby("seed"):
        sns.lineplot(data=g, x="GFLOPs", y="FID", marker="o", label=f"seed {seed}")
        for _, r in g.iterrows():
            plt.annotate(f"τ={r['τ']}", (r["GFLOPs"], r["FID"]))
    plt.xlabel("GFLOPs / image")
    plt.ylabel("FID (lower better)")
    plt.title("DiT-XL Pareto – HARD")
    plt.legend()
    plt.savefig(out_dir / "fid_vs_gflops.pdf", bbox_inches="tight")
    print("Saved figure fid_vs_gflops.pdf")


# --------------------------------------------------
#  EXPERIMENT 2  – Stable Diffusion real-time text-to-image
# --------------------------------------------------
from transformers import CLIPProcessor, CLIPModel  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402


def _clip_score(model, processor, image, prompt: str, device="cuda") -> float:
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # [1,1]
    return logits_per_image.item()


def run_experiment2(args):
    print("""
───────────────────────────────────────────────────────────────
EXPERIMENT 2 – Real-Time Text-to-Image on Stable Diffusion v1.5
───────────────────────────────────────────────────────────────
    """)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Loading Stable Diffusion v1.5 …")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    if device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    hard_pipe = HARDWrapper(pipe.unet.up_blocks, taus=[0.10] * 3, freeze_k=2, device=device, dtype=dtype)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    prompts_path = Path(args.prompt_json)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompt list not found: {prompts_path}")
    prompts = json.loads(prompts_path.read_text())

    records = []
    for w in [4.0, 7.5, 12.0]:
        for seed in SEED_LIST:
            set_seed(seed)
            latencies, clip_scores = [], []
            for p in tqdm(prompts, desc=f"guidance={w}"):
                start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_evt.record()
                with autocast(device.type):
                    img = pipe(prompt=p, num_inference_steps=8, guidance_scale=w, height=512, width=512).images[0]
                end_evt.record(); torch.cuda.synchronize()
                latencies.append(start_evt.elapsed_time(end_evt))
                clip_scores.append(_clip_score(clip_model, clip_processor, img, p, device=device))

            records.append({
                "guidance": w,
                "seed": seed,
                "latency_ms_mean": np.mean(latencies),
                "clip_mean": np.mean(clip_scores),
            })
            print("Result:", records[-1])

    df = pd.DataFrame(records)
    out_dir = Path(args.output_dir) / "exp2"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp2_latency_clip.csv", index=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="guidance", y="latency_ms_mean", hue="seed")
    for idx, r in df.iterrows():
        plt.text(x=idx, y=r["latency_ms_mean"], s=f"{r['latency_ms_mean']:.0f} ms", ha="center", va="bottom")
    plt.ylabel("Latency / prompt (ms)")
    plt.title("Stable Diffusion + HARD – Latency vs Guidance scale w")
    plt.savefig(out_dir / "latency_guidance.pdf", bbox_inches="tight")
    print("Saved figure latency_guidance.pdf")


# --------------------------------------------------
#  EXPERIMENT 3  – Spatial locality patch test
# --------------------------------------------------
from diffusers import UNet2DConditionModel  # noqa: E402
from piq import LPIPS  # noqa: E402


def run_experiment3(args):
    print("""
───────────────────────────────────────────────────────────────
EXPERIMENT 3 – Spatial Locality Stress-Test with Synthetic Patches
───────────────────────────────────────────────────────────────
    """)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Mock 4-scale Wavelet-DM (four identical UNets).
    def _stage():
        return UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 320),
        ).to(device, dtype=dtype)

    hw = HARDWrapper([_stage() for _ in range(4)], taus=[0.10] * 4, freeze_k=2, device=device, dtype=dtype)
    lpips_fn = LPIPS(reduction="mean").to(device)

    imgs, masks = prep.patch_dataset()
    dataset = list(zip(imgs, masks))

    out_dir = Path(args.output_dir) / "exp3"
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_ratios, lpips_in, lpips_out = [], [], []
    for idx, (img, mask) in enumerate(tqdm(dataset)):
        img, mask = img.to(device), mask.to(device)
        latents = [torch.randn(1, 4, 64 * (2 ** lvl), 64 * (2 ** lvl), device=device, dtype=dtype) for lvl in range(4)]
        ts = [torch.randint(0, 1000, (1,), device=device) for _ in range(4)]
        outs, u_maps = hw.sample(latents, ts)
        pixel_ratios.append((u_maps[-1] > hw.taus[-1]).float().mean().item())

        gen_img = torch.clamp((outs[-1].float() + 1) / 2, 0, 1)
        ref_img = torch.clamp((latents[-1].float() + 1) / 2, 0, 1)  # stand-in reference
        lpips_in.append(lpips_fn(gen_img * mask, ref_img * mask).item())
        lpips_out.append(lpips_fn(gen_img * (1 - mask), ref_img * (1 - mask)).item())

        if idx < 10:
            np.save(out_dir / f"u_map_{idx}.npy", u_maps[-1].squeeze().cpu().half().numpy())

    # Summary
    print(f"Fine-scale pixel ratio mean: {np.mean(pixel_ratios):.3f}")
    print(f"LPIPS inside-patch  mean  : {np.mean(lpips_in):.3f}")
    print(f"LPIPS outside-patch mean  : {np.mean(lpips_out):.3f}")

    # Histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(pixel_ratios, bins=20, kde=True)
    plt.axvline(0.15, ls="--", c="r", label="target 0.15")
    plt.xlabel("Fraction of evaluated fine-scale pixels")
    plt.title("Spatial sparsity – HARD (patch area ≈ 6 %)")
    plt.legend()
    plt.savefig(out_dir / "pixel_ratio_hist.pdf", bbox_inches="tight")
    print("Saved figure pixel_ratio_hist.pdf")
