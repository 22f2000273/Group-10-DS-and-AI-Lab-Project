#!/usr/bin/env python3
"""Pipeline tests: param count, output shape, augmentation pairing, no-aug determinism, loss."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import numpy as np
import torch

from src.data.dataset import TileDataset, _augment_pair  # noqa: E402
from src.models.model import UNet, count_params  # noqa: E402
from src.evaluation.train import BCEDice  # noqa: E402


def test_unet_param_count_and_output_shape():
    model = UNet()
    n = count_params(model)
    assert n == 31_037_633, f"param count {n:,} != 31,037,633"
    out = model(torch.randn(2, 3, 512, 512))
    assert tuple(out.shape) == (2, 1, 512, 512), f"output shape {tuple(out.shape)}"


def test_augmentation_same_transform_on_image_and_mask():
    # build an image whose 3 channels EQUAL the mask -> after augmentation they must stay equal
    msk = torch.zeros(1, 8, 8)
    msk[0, :4, :4] = 1.0
    img = msk.repeat(3, 1, 1)  # 3x8x8, channels == mask pattern

    torch.manual_seed(0)
    for trial in range(20):
        img_a, msk_a = _augment_pair(img.clone(), msk.clone())
        assert torch.equal(img_a[0], msk_a[0]), f"channel 0 != mask after augment (trial {trial})"
        assert torch.equal(img_a[1], img_a[2]), "channels diverged after augment"


def test_validation_never_augmented():
    # synthetic 2-tile dataset in the expected layout
    from PIL import Image
    import shutil

    root = Path(__file__).resolve().parents[1] / "_tmp_valcheck"
    d = root / "WHU" / "validation"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(2):
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        msk = np.zeros((512, 512), dtype=np.uint8)
        msk[:100, :100] = 1
        Image.fromarray(img).save(d / "images" / f"t{i}.png")
        Image.fromarray(msk).save(d / "masks" / f"t{i}.png")

    ds = TileDataset(root, "whu", "validation", augment=True)  # augment ignored for non-train
    a = ds[0]
    b = ds[0]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["mask"], b["mask"])
    expected_mask = torch.from_numpy(
        np.pad(np.ones((100, 100)), ((0, 412), (0, 412)))
    ).float()[None]
    assert torch.equal(a["mask"], expected_mask)

    # cleanup
    shutil.rmtree(root)


def test_bcedice_finite_and_backward():
    torch.manual_seed(0)
    crit = BCEDice()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = (torch.rand(2, 1, 64, 64) > 0.7).float()
    loss = crit(logits, target)
    assert torch.isfinite(loss), "BCEDice returned non-finite loss"
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


if __name__ == "__main__":
    import shutil  # noqa: F401
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
