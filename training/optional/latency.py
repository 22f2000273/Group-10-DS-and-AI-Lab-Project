#!/usr/bin/env python3
"""Inference latency measurement (measurement only; no training).

Loads the three seed-42 BASELINE best checkpoints, times prediction
(forward pass + threshold, batch size 1, 512x512) over up to 100 test
tiles of each dataset's own test split. Saves results to latency.md.
"""
import statistics as st
import time
from pathlib import Path

import torch

sys_path = str(Path(__file__).resolve().parents[2])
import sys
sys.path.insert(0, sys_path)

from src.data.dataset import TileDataset, make_loader
from src.models.model import build_model

CKPTS = {
    "whu": "runs/baseline/unet_whu_seed42/best.pt",
    "inria": "runs/baseline/unet_inria_seed42/best.pt",
    "mass": "runs/baseline/unet_mass_seed42/best.pt",
}
DEVICE = torch.device("cuda")
BATCH = 1
WARMUP = 10
N_TIMED = 100

results = {}
for ds, ckpt_path in CKPTS.items():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model("unet").to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds_obj = TileDataset("/home/bcilab/DSAI/data_view", ds, "test", augment=False)
    loader = make_loader(ds_obj, batch_size=BATCH, shuffle=False, num_workers=0, seed=0)
    times = []
    n_seen = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img = batch["image"].to(DEVICE)
            if i < WARMUP:
                logits = model(img)
                _ = torch.sigmoid(logits) > 0.5
                continue
            if i >= WARMUP + N_TIMED:
                break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(img)
            _ = torch.sigmoid(logits) > 0.5
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0 / BATCH)
            n_seen += 1
    mean_ms = st.mean(times)
    std_ms = st.stdev(times) if len(times) > 1 else 0.0
    results[ds] = (mean_ms, std_ms, n_seen)
    print(f"{ds}: {mean_ms:.2f} ms/tile +/- {std_ms:.2f} over {n_seen} tiles")

gpu_name = torch.cuda.get_device_name(0)
lines = [
    "# Inference latency measurement (no training)",
    "",
    f"GPU: {gpu_name}  |  batch size: {BATCH}  |  tile size: 512x512",
    f"Timed tiles per dataset: up to {N_TIMED} (Mass test split has only 40 tiles, so 30 timed)",
    f"Warm-up tiles: {WARMUP}  |  measured: forward pass + 0.5 threshold per tile",
    "",
    "| dataset | checkpoints used | tiles timed | mean ms/tile | std ms/tile | tiles/sec |",
    "|---|---|---|---:|---:|---:|",
]
for ds in ("whu", "inria", "mass"):
    mean_ms, std_ms, n = results[ds]
    lines.append(f"| {ds} | baseline unet seed-42 best.pt | {n} | {mean_ms:.2f} | {std_ms:.2f} | {1000.0/mean_ms:.1f} |")
lines.append("")
lines.append("Interpretation: per-tile forward+threshold on a single 512x512 tile on one L40S GPU.")
Path("training/optional/latency.md").write_text("\n".join(lines) + "\n")
print("saved training/optional/latency.md")
