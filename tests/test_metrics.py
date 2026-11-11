#!/usr/bin/env python3
"""Metric unit tests (plain asserts, pytest-compatible)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import torch

from src.evaluation.metrics import ConfusionAccumulator  # noqa: E402
from src.evaluation.train import dice_loss  # noqa: E402


def _logits(pred01):
    """Map a 0/1 pattern to strong logits (+/-8)."""
    t = torch.tensor(pred01, dtype=torch.float32)
    return (t * 2 - 1) * 8


def test_perfect_prediction_iou_one():
    acc = ConfusionAccumulator()
    target = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    acc.update(_logits([[[[1, 0], [1, 1]]]]), target)
    m = acc.compute()
    assert m["iou"] == 1.0 and m["f1"] == 1.0


def test_all_background_prediction_on_nonempty_target_iou_zero():
    acc = ConfusionAccumulator()
    target = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    acc.update(_logits([[[[0, 0], [0, 0]]]]), target)
    m = acc.compute()
    assert m["iou"] == 0.0 and m["tp"] == 0 and m["fn"] == 3


def test_hand_computed_example():
    # pred 1,1,0,0 ; target 1,0,1,0 -> tp=1 fp=1 fn=1
    acc = ConfusionAccumulator()
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    acc.update(_logits([[[[1, 1], [0, 0]]]]), target)
    m = acc.compute()
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert abs(m["iou"] - 1 / 3) < 1e-6
    assert abs(m["precision"] - 0.5) < 1e-6
    assert abs(m["recall"] - 0.5) < 1e-6
    assert abs(m["f1"] - 0.5) < 1e-6


def test_accumulation_equals_one_big_batch():
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 2, 2)
    target = (torch.rand(4, 1, 2, 2) > 0.5).float()

    acc_parts = ConfusionAccumulator()
    acc_parts.update(logits[:2], target[:2])
    acc_parts.update(logits[2:], target[2:])

    acc_big = ConfusionAccumulator()
    acc_big.update(logits, target)

    assert acc_parts.compute() == acc_big.compute()


def test_empty_target_empty_prediction_no_nan():
    acc = ConfusionAccumulator()
    target = torch.zeros(1, 1, 2, 2)
    acc.update(_logits([[[[0, 0], [0, 0]]]]), target)
    m = acc.compute()
    for k in ("iou", "precision", "recall", "f1"):
        assert m[k] == m[k]  # not NaN
        assert m[k] == 0.0
    assert m["tn"] == 4


def test_dice_loss_all_background_prediction_near_one():
    logits = torch.full((1, 1, 4, 4), -8.0)  # ~all background
    target = torch.ones(1, 1, 4, 4)  # fully non-empty target
    loss = dice_loss(logits, target)
    # with eps=1: loss = 1 - (0 + 1) / (0 + 16 + 1) ~= 0.941
    assert loss.item() > 0.9, f"dice_loss should be close to 1, got {loss.item()}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
