#!/usr/bin/env python3
"""Segmentation metrics accumulated over whole evaluation sets.

Threshold fixed at 0.5. Reports IoU / precision / recall / F1 plus raw counts.
No pixel accuracy, no boundary/object metrics (later milestones).
"""
from __future__ import annotations

from typing import Dict

import torch


class ConfusionAccumulator:
    """Accumulates TP/FP/FN/TN over ALL pixels of an evaluation set (int64)."""

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.n_pixels = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        """logits: (B,1,H,W) raw model outputs; target: (B,1,H,W) in {0,1}."""
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > threshold)
            tgt = (target > 0)

            pred = pred.reshape(-1)
            tgt = tgt.reshape(-1)

            self.tp += int((pred & tgt).sum().item())
            self.fp += int((pred & ~tgt).sum().item())
            self.fn += int((~pred & tgt).sum().item())
            self.tn += int((~pred & ~tgt).sum().item())
            self.n_pixels += int(pred.numel())

    def compute(self) -> Dict[str, float | int]:
        """Returns metrics dict; 0.0 for zero denominators (never NaN)."""
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(self.tn),
            "n_pixels": int(self.n_pixels),
        }


if __name__ == "__main__":
    # tiny self-check
    acc = ConfusionAccumulator()
    logits = torch.tensor([[[[5.0, -5.0], [5.0, -5.0]]]])       # pred 1,0,1,0
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])          # true 1,1,0,0
    acc.update(logits, target)
    print(acc.compute())  # tp=1 fp=1 fn=1 -> iou=1/3
