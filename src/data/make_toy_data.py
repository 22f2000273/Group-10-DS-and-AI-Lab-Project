#!/usr/bin/env python3
"""Generate a tiny SYNTHETIC dataset in the exact expected layout (test helper only).

Images: textured noise background + random bright rectangles as 'buildings'.
Masks: the rectangles as 0/1. A few empty tiles. Different colour statistic per dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# per-dataset background colour statistics (mean RGB) so the toy data differs across datasets
STYLE = {
    "WHU":   {"mean": (110, 113, 105), "spread": 25, "building": (200, 210, 200)},
    "Inria": {"mean": (100, 107, 98),  "spread": 22, "building": (190, 195, 185)},
    "Mass":  {"mean": (92, 94, 86),    "spread": 20, "building": (205, 205, 195)},
}


def make_tile(rng: np.random.Generator, style: dict, size: int) -> tuple[np.ndarray, np.ndarray]:
    bg = rng.normal(0, 1, (size, size, 3)) * style["spread"] + np.array(style["mean"])
    img = np.clip(bg, 0, 255).astype(np.uint8)
    msk = np.zeros((size, size), dtype=np.uint8)

    n_rects = rng.integers(0, 5)  # 0-4 rectangles (0 -> empty tile)
    for _ in range(n_rects):
        h, w = rng.integers(size // 10, size // 3, size=2)
        y, x = rng.integers(0, size - h), rng.integers(0, size - w)
        colour = np.clip(np.array(style["building"]) + rng.normal(0, 10, 3), 0, 255)
        img[y:y + h, x:x + w] = colour.astype(np.uint8)
        msk[y:y + h, x:x + w] = 1
    return img, msk


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate tiny synthetic dataset (TEST HELPER)")
    p.add_argument("--out", required=True)
    p.add_argument("--n_train", type=int, default=16)
    p.add_argument("--n_val", type=int, default=4)
    p.add_argument("--n_test", type=int, default=4)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out = Path(args.out)
    for ds_folder in ("WHU", "Inria", "Mass"):
        counts = {"train": args.n_train, "validation": args.n_val, "test": args.n_test}
        for split, n in counts.items():
            rng = np.random.default_rng(args.seed + hash((ds_folder, split)) % 10000)
            idir = out / ds_folder / split / "images"
            mdir = out / ds_folder / split / "masks"
            idir.mkdir(parents=True, exist_ok=True)
            mdir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img, msk = make_tile(rng, STYLE[ds_folder], args.size)
                name = f"{ds_folder.lower()}_{ds_folder.lower()}src_{i:03d}.png"
                Image.fromarray(img).save(idir / name)
                Image.fromarray(msk).save(mdir / name)
        print(f"{ds_folder}: {counts} tiles written")

    print(f"toy dataset ready at {out}")


if __name__ == "__main__":
    main()
