#!/usr/bin/env python3
"""Tile dataset for prepared 512x512 rooftop segmentation tiles.

Reads from:
    <data_root>/WHU/{train,validation,test}/{images,masks}/*.png
    <data_root>/Inria/{train,validation,test}/{images,masks}/*.png
    <data_root>/Mass/{train,validation,test}/{images,masks}/*.png

Keys: whu -> WHU, inria -> Inria, mass -> Mass.
Images: 3xHxW float32 in [0,1] (just /255, nothing else). Masks: 1xHxW {0,1}.
"""
from __future__ import annotations

import functools
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

FOLDER_OF_KEY = {"whu": "WHU", "inria": "Inria", "mass": "Mass"}
SPLITS = ("train", "validation", "test")


def _pair_paths(images_dir, masks_dir):
    """Sorted (image, mask) pairs; fail loudly if a mask partner is missing."""
    pairs = []
    for img in sorted(images_dir.glob("*.png")):
        msk = masks_dir / img.name
        if not msk.exists():
            raise FileNotFoundError(f"missing mask partner for {img} (looked for {msk})")
        pairs.append((img, msk))
    if not pairs:
        raise FileNotFoundError(f"no PNG tiles found in {images_dir}")
    return pairs


def _augment_pair(img, msk):
    """Random flips / rot90, applied identically to image and mask."""
    if random.random() < 0.5:
        img = torch.flip(img, dims=[-1])
        msk = torch.flip(msk, dims=[-1])
    if random.random() < 0.5:
        img = torch.flip(img, dims=[-2])
        msk = torch.flip(msk, dims=[-2])
    k = random.choice((0, 1, 2, 3))
    if k:
        img = torch.rot90(img, k, dims=[-2, -1])
        msk = torch.rot90(msk, k, dims=[-2, -1])
    return img, msk


class TileDataset(Dataset):
    """Prepared rooftop tiles. Validation/test are never augmented or cropped."""

    def __init__(self, data_root, dataset, split, augment=False, crop=None, max_tiles=None):
        if dataset not in FOLDER_OF_KEY:
            raise ValueError(f"unknown dataset '{dataset}', expected one of {sorted(FOLDER_OF_KEY)}")
        if split not in SPLITS:
            raise ValueError(f"unknown split '{split}', expected one of {SPLITS}")

        root = Path(data_root) / FOLDER_OF_KEY[dataset] / split
        images_dir, masks_dir = root / "images", root / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise FileNotFoundError(
                f"dataset folder not found: need {images_dir} and {masks_dir}"
            )

        self.pairs = _pair_paths(images_dir, masks_dir)
        if max_tiles is not None:
            self.pairs = self.pairs[:int(max_tiles)]

        self.split = split
        self.augment = bool(augment) and split == "train"
        self.crop = crop
        if self.augment and crop is not None:
            if crop % 16 != 0:
                raise ValueError(f"crop must divide by 16, got {crop}")
            if crop > 512:
                raise ValueError(f"crop must be <= 512, got {crop}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        msk = np.asarray(Image.open(msk_path), dtype=np.float32)

        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy())   # 3xHxW
        msk_t = torch.from_numpy((msk > 0).astype(np.float32)[None])  # 1xHxW

        if self.augment:
            img_t, msk_t = _augment_pair(img_t, msk_t)
            if self.crop is not None and self.crop < img_t.shape[-1]:
                top = random.randint(0, img_t.shape[-1] - self.crop)
                left = random.randint(0, img_t.shape[-1] - self.crop)
                img_t = img_t[..., top:top + self.crop, left:left + self.crop]
                msk_t = msk_t[..., top:top + self.crop, left:left + self.crop]

        return {"image": img_t.contiguous(), "mask": msk_t.contiguous(), "name": img_path.stem}


def _worker_init_fn(seed, worker_id):
    """Module-level worker init: seed python/numpy/torch per worker (picklable)."""
    s = seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_loader(dataset, batch_size=8, shuffle=False, num_workers=2, seed=0):
    """DataLoader with a seeded generator + worker_init_fn (reproducible augmentation)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=gen,
        worker_init_fn=functools.partial(_worker_init_fn, seed),
        pin_memory=torch.cuda.is_available(),
    )


def count_tiles(data_root):
    """Counts per dataset/split from the folder scan (source of truth)."""
    out = {}
    for key, folder in FOLDER_OF_KEY.items():
        out[key] = {}
        for split in SPLITS:
            d = Path(data_root) / folder / split / "images"
            out[key][split] = len(list(d.glob("*.png"))) if d.is_dir() else 0
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print tile counts per dataset/split")
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()
    print(count_tiles(args.data_root))
