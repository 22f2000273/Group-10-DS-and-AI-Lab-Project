#!/usr/bin/env python3
"""Acceptance checks on the prepared (read-only) dataset.

Exits non-zero on any failure. Folder scan is the source of truth;
metadata.csv is only cross-checked (row count == tiles) if present.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import torch
from PIL import Image

from src.data.dataset import FOLDER_OF_KEY, SPLITS, count_tiles
from src.models.model import UNet, count_params

EXPECTED = {
    "whu":   {"train": 4736, "validation": 1036, "test": 2416},   # total 8188
    "inria": {"train": 10530, "validation": 2025, "test": 2025},  # total 14580
    "mass":  {"train": 548,  "validation": 16,   "test": 40},    # total 604
}
GRAND_TOTAL = 23372


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures += 1
        return ok


def scan_pairs(root: Path, full: bool, sample: int) -> tuple[list[tuple[Path, Path]], list[str]]:
    pairs: list[tuple[Path, Path]] = []
    problems: list[str] = []
    for key, folder in FOLDER_OF_KEY.items():
        for split in SPLITS:
            imgs = sorted((root / folder / split / "images").glob("*.png"))
            for img in (imgs if full else random.sample(imgs, min(sample, len(imgs)))):
                msk = root / folder / split / "masks" / img.name
                if msk.exists():
                    pairs.append((img, msk))
                else:
                    problems.append(f"missing mask partner: {img}")
    return pairs, problems


def verify_tile(img_path: Path, msk_path: Path) -> str | None:
    img = np.asarray(Image.open(img_path))
    msk = np.asarray(Image.open(msk_path))
    msk_hw = msk.shape[:2] if msk.ndim == 3 else msk.shape
    if img.shape[:2] != (512, 512) or msk_hw != (512, 512):
        return f"wrong size: {img_path.name} img={img.shape[:2]} mask={msk_hw}"
    if img.ndim != 3 or img.shape[2] != 3:
        return f"image not 3-channel: {img_path} shape={img.shape}"
    uniq = set(np.unique(msk).tolist())
    if not uniq.issubset({0, 1}):
        return f"mask values not {{0,1}}: {msk_path} has {sorted(uniq)[:6]}"
    return None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Check the prepared dataset")
    p.add_argument("--data_root", required=True)
    p.add_argument("--full", action="store_true", help="scan ALL tiles (default: 200 per split)")
    p.add_argument("--toy", action="store_true", help="toy data: skip the exact-count comparison")
    args = p.parse_args(argv)

    random.seed(0)
    root = Path(args.data_root)
    rep = Report()

    # 1. all 9 dataset/split folders with images/ and masks/
    missing = []
    for key, folder in FOLDER_OF_KEY.items():
        for split in SPLITS:
            for sub in ("images", "masks"):
                d = root / folder / split / sub
                if not d.is_dir():
                    missing.append(str(d))
    rep.check("1. all 9 dataset/split folders exist (images/ + masks/)",
              not missing, ", ".join(missing) if missing else "")

    # 2. exact counts
    if args.toy:
        print("[SKIP] 2. exact counts (--toy mode)")
    else:
        counts = count_tiles(root)
        bad = []
        for key, exp in EXPECTED.items():
            for split, n in exp.items():
                actual = counts[key][split]
                if actual != n:
                    bad.append(f"{key}/{split}: {actual} != {n}")
        total = sum(sum(v.values()) for v in counts.values())
        rep.check("2. tile counts match expected (WHU 8188, Inria 14580, Mass 604 = 23372)",
                  not bad, "; ".join(bad) if bad else f"total={total}")

    # 3. pairing / sizes / channels / mask values
    pairs, problems = scan_pairs(root, args.full, 200)
    problems += [msg for msg in (verify_tile(i, m) for i, m in pairs) if msg]
    rep.check(f"3. pairing+size+channels+mask values on {len(pairs)} tiles",
              not problems, "; ".join(problems[:5]) if problems else "")

    # 4. metadata.csv row count cross-check (warn only if absent)
    meta = root / "metadata.csv"
    if meta.exists():
        import csv
        with open(meta) as f:
            rows = sum(1 for _ in f) - 1
        total = sum(sum(v.values()) for v in count_tiles(root).values())
        rep.check("4. metadata.csv row count == total tiles", rows == total,
                  f"rows={rows}, tiles={total}")
    else:
        print(f"[WARN] 4. metadata.csv not present at {meta} (ok — folder scan is the source of truth)")

    # 5. forward pass + parameter count
    try:
        model = UNet()
        n_params = count_params(model)
        batch = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = model(batch)
        rep.check("5. UNet forward pass shape (2,1,512,512)", tuple(out.shape) == (2, 1, 512, 512),
                  f"got {tuple(out.shape)}")
        rep.check("   UNet param count == 31,037,633", n_params == 31_037_633,
                  f"counted {n_params:,}")
    except Exception as e:  # pragma: no cover
        rep.check("5. UNet forward pass + param count", False, f"error: {e}")

    print("\nRESULT:", "ALL CHECKS PASSED" if rep.failures == 0 else f"{rep.failures} CHECK(S) FAILED")
    sys.exit(0 if rep.failures == 0 else 1)


if __name__ == "__main__":
    main()
