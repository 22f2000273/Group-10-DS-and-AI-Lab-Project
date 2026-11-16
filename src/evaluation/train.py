#!/usr/bin/env python3
"""Train the baseline U-Net on ONE dataset, then test on all three.

Defaults here are SMOKE-TEST placeholders only — final hyper-parameters are
fixed in Milestone 4. Do not tune here.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from src.data.dataset import FOLDER_OF_KEY, TileDataset, make_loader
from src.evaluation.evaluate import evaluate_model
from src.evaluation.metrics import ConfusionAccumulator
from src.models.model import build_model, count_params

ALL_DATASETS = ("whu", "inria", "mass")


def dice_loss(logits, target, eps=1.0):
    """Soft Dice loss on logits (sigmoid applied here)."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    denom = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


class BCEDice(nn.Module):
    """BCEWithLogits + soft Dice, unweighted; empty tiles stay in training."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return self.bce(logits, target) + dice_loss(logits, target)


class FocalDice(nn.Module):
    """Focal loss (gamma=2.0 on sigmoid probabilities, averaged over all batch
    pixels) + the same soft Dice term as BCEDice."""

    def __init__(self, gamma=2.0, eps=1.0):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = -(target * torch.log(p + self.eps) + (1 - target) * torch.log(1 - p + self.eps))
        pt = p * target + (1 - p) * (1 - target)
        focal = ((1 - pt) ** self.gamma * ce).mean()
        return focal + dice_loss(logits, target)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    return dev


def make_args_parser():
    p = argparse.ArgumentParser(description="Train baseline U-Net on one dataset")
    p.add_argument("--data_root", required=True)
    p.add_argument("--train_set", required=True, choices=ALL_DATASETS)
    p.add_argument("--out", required=True)
    p.add_argument("--arch", default="unet", choices=["unet", "smp_unet"])
    p.add_argument("--pretrained", action="store_true",
                   help="smp only: use ImageNet encoder weights (the only allowed download)")
    p.add_argument("--epochs", type=int, default=2,
                   help="PLACEHOLDER default for smoke tests; final values fixed in Milestone 4")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="none", choices=["none", "cosine", "step"])
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--crop", type=int, default=None, help="train-time random crop size (multiple of 16)")
    p.add_argument("--amp", action="store_true", help="mixed precision on cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None, help="path to last.pt to resume")
    p.add_argument("--dry_run", action="store_true",
                   help="tiny settings (8 tiles/split, 2 epochs, batch 2, crop 256) but the FULL flow")
    p.add_argument("--max_steps", type=int, default=None,
                   help="stop training after exactly N optimiser steps, counted across epochs "
                        "(equal-budget experiments); scheduler steps per optimiser step when set")
    p.add_argument("--subset_size", type=int, default=None,
                   help="train on a random subset of N tiles from the training split only "
                        "(Phase 5 subset control)")
    p.add_argument("--subset_seed", type=int, default=42,
                   help="seed for the --subset_size draw (chosen tile names are written to "
                        "subset_tiles.txt in the run folder)")
    p.add_argument("--loss", default="bce_dice", choices=["bce_dice", "focal_dice"],
                   help="loss function: 'bce_dice' (baseline) or 'focal_dice' "
                        "(focal loss gamma 2.0 + same Dice term)")
    return p


def build_optimizer(name, model, lr, wd):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"unknown optimizer {name}")


def build_scheduler(name, opt, epochs, steps=None):
    """When `steps` (max_steps) is given, the schedule is built per optimiser
    step instead of per epoch, so it covers the whole step budget."""
    if name == "none":
        return None
    if name == "cosine":
        if steps is not None:
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if name == "step":
        if steps is not None:
            return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 3), gamma=0.1)
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.1)
    raise ValueError(f"unknown scheduler {name}")


@torch.no_grad()
def quick_val(model, loader, device):
    """Validation loss + IoU (shared accumulation path)."""
    model.eval()
    criterion = BCEDice()
    acc = ConfusionAccumulator()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        img = batch["image"].to(device)
        msk = batch["mask"].to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(img)
            loss = criterion(logits, msk)
        acc.update(logits.float(), msk)
        total_loss += float(loss.item())
        n_batches += 1
    m = acc.compute()
    return total_loss / max(1, n_batches), m["iou"], m["f1"]


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_iou,
                    patience_ctr, seed, step):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "best_val_iou": best_iou,
        "patience_counter": patience_ctr,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "seed": seed,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def main(argv=None):
    args = make_args_parser().parse_args(argv)

    if args.dry_run:
        print("[dry_run] forcing tiny settings: max 8 tiles/split, 2 epochs, batch 2, crop 256")
        args.epochs, args.batch_size, args.crop = 2, 2, (args.crop or 256)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = get_device()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.arch, pretrained=args.pretrained).to(device)
    params = count_params(model)

    config = {
        **vars(args),
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "param_count": params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "hyper-parameter defaults are smoke-test placeholders; fixed in Milestone 4",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"params: {params:,}")

    train_ds = TileDataset(args.data_root, args.train_set, "train",
                           augment=True, crop=args.crop,
                           max_tiles=8 if args.dry_run else None)
    if args.subset_size:
        full_n = len(train_ds)
        n = min(args.subset_size, full_n)
        rng = random.Random(args.subset_seed)
        idx = sorted(rng.sample(range(full_n), n))
        (out / "subset_tiles.txt").write_text(
            "\n".join(train_ds.pairs[i][0].name for i in idx) + "\n")
        train_ds = Subset(train_ds, idx)
        print(f"subset: using {n} of {full_n} training tiles "
              f"(subset_seed {args.subset_seed}) -> subset_tiles.txt")
    val_ds = TileDataset(args.data_root, args.train_set, "validation",
                         max_tiles=8 if args.dry_run else None)
    train_ld = make_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, seed=args.seed)
    val_ld = make_loader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, seed=args.seed)
    print(f"train tiles: {len(train_ds)}, validation tiles: {len(val_ds)}")

    if args.loss == "focal_dice":
        criterion = FocalDice()
    else:
        criterion = BCEDice()
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs, steps=args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    start_epoch, best_iou, patience_ctr = 1, -1.0, 0
    start_step = 0
    if args.resume:
        if not Path(args.resume).exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        start_step = ckpt.get("step", 0)
        best_iou = ckpt["best_val_iou"]
        patience_ctr = ckpt["patience_counter"]
        if ckpt.get("rng", {}).get("python"):
            random.setstate(ckpt["rng"]["python"])
            np.random.set_state(ckpt["rng"]["numpy"])
            torch.set_rng_state(ckpt["rng"]["torch"])
            if torch.cuda.is_available() and ckpt["rng"]["cuda"]:
                torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        print(f"resumed from {args.resume}: next epoch {start_epoch}, step {start_step}, best val IoU {best_iou:.4f}")

    log_rows = []
    total_steps = start_step
    stopped_by_steps = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, total_loss, n_batches = time.time(), 0.0, 0
        for batch in train_ld:
            img = batch["image"].to(device, non_blocking=True)
            msk = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(scaler.is_enabled())):
                logits = model(img)
                loss = criterion(logits, msk)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_steps += 1
            total_loss += float(loss.item())
            n_batches += 1
            # with a step budget the scheduler advances per optimiser step
            if scheduler and args.max_steps:
                scheduler.step()
            if args.max_steps and total_steps >= args.max_steps:
                stopped_by_steps = True
                break
        if scheduler and not args.max_steps:
            scheduler.step()

        val_loss, val_iou, val_f1 = quick_val(model, val_ld, device)
        lr_now = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0
        print(f"epoch {epoch}/{args.epochs} | step {total_steps} | train_loss {total_loss/max(1,n_batches):.4f} | "
              f"val_loss {val_loss:.4f} | val_IoU {val_iou:.4f} | val_F1 {val_f1:.4f} | "
              f"lr {lr_now:.2e} | {secs:.1f}s")
        log_rows.append([epoch, total_steps, f"{total_loss/max(1,n_batches):.6f}",
                         f"{val_loss:.6f}", f"{val_iou:.6f}", f"{val_f1:.6f}",
                         f"{lr_now:.2e}", f"{secs:.2f}"])

        save_checkpoint(out / "last.pt", model, optimizer, scheduler, scaler,
                        epoch, max(best_iou, val_iou), patience_ctr, args.seed, total_steps)
        if val_iou > best_iou:
            best_iou, patience_ctr = val_iou, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_iou": val_iou},
                       out / "best.pt")
            print(f"  -> best.pt saved (val IoU {val_iou:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"early stopping at epoch {epoch} (no val-IoU gain for {args.patience} epochs)")
                break
        if stopped_by_steps:
            print(f"max_steps {args.max_steps} reached after {total_steps} optimiser steps "
                  f"(stopped mid-training at epoch {epoch})")
            break

    with open(out / "log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "train_loss", "val_loss", "val_iou", "val_f1", "lr", "seconds"])
        w.writerows(log_rows)

    # record the number of optimiser steps actually completed in config.json
    config["steps_completed"] = total_steps
    (out / "config.json").write_text(json.dumps(config, indent=2))

    best_path = out / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        best_epoch = state["epoch"]
    else:
        print("WARNING: best.pt missing (val IoU never improved?) — evaluating last state")
        best_epoch = args.epochs

    test_results = {}
    for ds in ALL_DATASETS:
        test_dir = Path(args.data_root) / FOLDER_OF_KEY[ds] / "test" / "images"
        if not test_dir.is_dir():
            print(f"WARNING: test folder for '{ds}' missing at {test_dir} — skipping (not silently)")
            continue
        ds_obj = TileDataset(args.data_root, ds, "test",
                             max_tiles=8 if args.dry_run else None)
        ld = make_loader(ds_obj, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, seed=args.seed)
        m = evaluate_model(model, ld, device)
        test_results[ds] = m
        print(f"TEST {ds}: IoU {m['iou']:.4f} F1 {m['f1']:.4f} (n={len(ds_obj)} tiles)")

    results = {
        "train_set": args.train_set,
        "arch": args.arch,
        "seed": args.seed,
        "param_count": params,
        "best_epoch": best_epoch,
        "best_val_iou": best_iou if best_iou >= 0 else None,
        "test": test_results,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"results.json written to {out/'results.json'}")


if __name__ == "__main__":
    main()
