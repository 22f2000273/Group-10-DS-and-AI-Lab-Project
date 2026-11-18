# Generates Section-9 training-progress figures (+ bonus comparison figures) for the M4 report.
# Data source: per-epoch log.csv inside each run folder. Nothing is re-run; existing files are not modified.
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
OUT = Path(__file__).resolve().parent

DS = ["mass", "whu", "inria"]
DS_TITLE = {"mass": "Massachusetts", "whu": "WHU", "inria": "Inria"}

def load(run_path):
    rows = list(csv.DictReader(open(Path(run_path) / "log.csv")))
    ep = [int(r["epoch"]) for r in rows]
    return dict(
        epoch=ep,
        train_loss=[float(r["train_loss"]) for r in rows],
        val_loss=[float(r["val_loss"]) for r in rows],
        val_iou=[float(r["val_iou"]) for r in rows],
    )

def best(d):
    return d["epoch"][max(range(len(d["val_iou"])), key=lambda i: d["val_iou"][i])]

# --- 1. per-baseline two-panel figures --------------------------------------
for ds in DS:
    d = load(RUNS / "baseline" / f"unet_{ds}_seed42")
    b = best(d)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(d["epoch"], d["train_loss"], label="Train loss")
    ax1.plot(d["epoch"], d["val_loss"], label="Validation loss (BCE+Dice)")
    ax1.axvline(b, ls="--", c="gray", label=f"Best epoch ({b})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title(f"{DS_TITLE[ds]} baseline (seed 42)")
    ax1.legend()
    ax2.plot(d["epoch"], d["val_iou"], c="tab:green")
    ax2.axvline(b, ls="--", c="gray")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation IoU"); ax2.set_ylim(0, 1)
    ax2.set_title(f"{DS_TITLE[ds]} — best val IoU {max(d['val_iou']):.4f} @ epoch {b}")
    fig.tight_layout()
    fig.savefig(OUT / f"curve_baseline_{ds}.png", dpi=150)
    plt.close(fig)

# --- 2. all baselines val IoU on one plot ------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for ds in DS:
    d = load(RUNS / "baseline" / f"unet_{ds}_seed42")
    ax.plot(d["epoch"], d["val_iou"], label=DS_TITLE[ds])
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation IoU"); ax.set_ylim(0, 1)
ax.set_title("Validation IoU vs epoch — three baselines (Protocol A, seed 42)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "curves_val_iou_all_baselines.png", dpi=150)
plt.close(fig)

# --- 3. seed repeats ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, ds in zip(axes, DS):
    for seed in [42, 43, 44]:
        base = RUNS / ("baseline" if seed == 42 else "seeds")
        d = load(base / f"unet_{ds}_seed{seed}")
        ax.plot(d["epoch"], d["val_iou"], label=f"seed {seed}")
    ax.set_xlabel("Epoch"); ax.set_title(DS_TITLE[ds])
axes[0].set_ylabel("Validation IoU"); axes[0].set_ylim(0, 1)
axes[0].legend()
fig.suptitle("Seed-repeat stability — validation IoU vs epoch (seeds 42/43/44)")
fig.tight_layout()
fig.savefig(OUT / "curves_seed_repeats.png", dpi=150)
plt.close(fig)

# --- 4. baseline cross-dataset IoU matrix heatmap ------------------------------
import json
m = [[None] * 3 for _ in range(3)]
for i, tr in enumerate(DS):
    t = json.load(open(RUNS / "baseline" / f"unet_{tr}_seed42" / "results.json"))["test"]
    for j, te in enumerate(DS):
        m[i][j] = t[te]["iou"]
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow([[v for v in row] for row in m], cmap="viridis", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{m[i][j]:.4f}", ha="center", va="center",
                color="white" if m[i][j] < 0.55 else "black", fontsize=11)
ax.set_xticks(range(3)); ax.set_xticklabels([DS_TITLE[d] for d in DS])
ax.set_yticks(range(3)); ax.set_yticklabels([f"trained on {DS_TITLE[d]}" for d in DS])
ax.set_title("Baseline cross-dataset test IoU matrix\n(trained rows x tested columns, seed 42)")
fig.colorbar(im, ax=ax, label="Test IoU")
fig.tight_layout()
fig.savefig(OUT / "matrix_baseline_heatmap.png", dpi=150)
plt.close(fig)

# --- 5. diagonal comparison bars: baseline vs variant vs fixed focal ----------
import re
def diagonal(run):
    t = json.load(open(RUNS / "optional" / run / "results.json"))["test"]
    return round(t[run.split("_")[-2] if False else re.search(r"(mass|whu|inria)", run).group(1)]["iou"], 4)
methods = {
    "Baseline (BCE+Dice)": {"mass": 0.6665, "whu": 0.8624, "inria": 0.7169},
    "ResNet-34 variant": {ds: diagonal(f"VARIANT_smp_unet_{ds}_seed42") for ds in DS},
    "Focal+Dice (fixed)": {ds: diagonal(f"FOCALFIX_{ds}_seed42") for ds in DS},
}
import numpy as np
x = np.arange(3); w = 0.25
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for k, (name, vals) in enumerate(methods.items()):
    ax.bar(x + (k - 1) * w, [vals[ds] for ds in DS], w, label=name)
ax.set_xticks(x); ax.set_xticklabels([DS_TITLE[d] for d in DS])
ax.set_ylabel("In-domain test IoU"); ax.set_ylim(0, 1)
ax.set_title("In-domain test IoU — baseline vs ResNet-34 variant vs fixed focal loss (seed 42)")
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "diagonal_comparison_bar.png", dpi=150)
plt.close(fig)

# --- 6. equal-budget curves (3,450 steps) -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, ds in zip(axes, ["whu", "inria"]):
    full = load(RUNS / "baseline" / f"unet_{ds}_seed42")
    bud = load(RUNS / "equal_budget" / f"unet_{ds}_steps3450")
    ax.plot(full["epoch"], full["val_iou"], label="Full budget (50 ep)")
    ax.plot(bud["epoch"], bud["val_iou"], label="Equal budget (3,450 steps)")
    ax.set_xlabel("Epoch"); ax.set_title(DS_TITLE[ds])
    steps_per_ep = 3450 / len(bud["epoch"])
    ax.axvline(len(bud["epoch"]), ls="--", c="gray")
    ax.annotate("3,450 steps", (len(bud["epoch"]), ax.get_ylim()[1] * 0.92),
                ha="right", fontsize=9, color="gray")
axes[0].set_ylabel("Validation IoU"); axes[0].set_ylim(0, 1); axes[0].legend()
fig.suptitle("Equal optimiser-step budget vs full budget — validation IoU (seed 42)")
fig.tight_layout()
fig.savefig(OUT / "curves_equal_budget.png", dpi=150)
plt.close(fig)

print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
