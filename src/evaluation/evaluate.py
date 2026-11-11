#!/usr/bin/env python3
"""Evaluate a checkpoint on a given split of one or more datasets.

Deterministic: model.eval(), no augmentation, no crop, threshold 0.5,
cudnn.benchmark OFF. Running twice on the same checkpoint gives identical numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import torch

from src.data.dataset import TileDataset, make_loader
from src.evaluation.metrics import ConfusionAccumulator


@torch.no_grad()
def evaluate_model(model, loader, device) -> dict:
    """Shared evaluation path: returns the metrics dict (accumulated over all pixels)."""
    was_training = model.training
    model.eval()
    acc = ConfusionAccumulator()
    for batch in loader:
        img = batch["image"].to(device)
        msk = batch["mask"].to(device)
        logits = model(img)
        acc.update(logits.float(), msk, threshold=0.5)
    if was_training:
        model.train()
    return acc.compute()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--datasets", nargs="+", default=["whu", "inria", "mass"],
                   choices=["whu", "inria", "mass"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", default=None, help="output JSON (default: <ckpt_dir>/eval_<split>.json)")
    args = p.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    cfg_path = ckpt_path.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found next to checkpoint: {cfg_path} "
                                f"(needed to rebuild the model)")
    cfg = json.loads(cfg_path.read_text())

    torch.backends.cudnn.benchmark = False  # determinism over speed
    torch.use_deterministic_algorithms(True)  # deterministic GPU ops
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from src.models.model import build_model  # local import keeps --help light
    model = build_model(cfg.get("arch", "unet"), pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    results = {}
    for ds in args.datasets:
        ds_obj = TileDataset(args.data_root, ds, args.split)
        ld = make_loader(ds_obj, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, seed=0)  # 0 workers = deterministic
        m = evaluate_model(model, ld, device)
        results[ds] = m

    print(f"\n== Evaluation | checkpoint {ckpt_path.name} | split {args.split} ==")
    print(f"{'dataset':<10} {'IoU':>8} {'F1':>8} {'prec':>8} {'recall':>8}")
    for ds, m in results.items():
        print(f"{ds:<10} {m['iou']:>8.4f} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}")

    out_path = Path(args.out) if args.out else ckpt_path.parent / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"checkpoint": str(ckpt_path), "split": args.split,
                                    "results": results}, indent=2))
    print(f"JSON written: {out_path}")


if __name__ == "__main__":
    main()
