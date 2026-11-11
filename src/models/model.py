#!/usr/bin/env python3
"""Plain U-Net (from scratch) plus optional segmentation_models_pytorch variant.

Outputs raw LOGITS (no sigmoid). Data pipeline stays plain 0-1; ImageNet
normalisation for smp happens inside forward.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv3x3 pad 1, bias=False -> BatchNorm2d -> ReLU, twice."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d(2) -> DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """ConvTranspose2d(x2) -> concat with skip -> DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    @staticmethod
    def _pad(x, skip):
        # pad x to skip's size (odd input sizes)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return x, skip

    def forward(self, x, skip):
        x = self.up(x)
        x, skip = self._pad(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Plain U-Net, 64 base channels, encoder 64..1024, head Conv1x1 -> logits."""

    def __init__(self, in_ch=3, n_classes=1, base=64):
        super().__init__()
        b = base
        self.inc = DoubleConv(in_ch, b)          # 64
        self.down1 = Down(b, b * 2)              # 128
        self.down2 = Down(b * 2, b * 4)          # 256
        self.down3 = Down(b * 4, b * 8)          # 512
        self.down4 = Down(b * 8, b * 16)         # 1024 (bottleneck)
        self.up1 = Up(b * 16, b * 8)             # 512
        self.up2 = Up(b * 8, b * 4)              # 256
        self.up3 = Up(b * 4, b * 2)              # 128
        self.up4 = Up(b * 2, b)                  # 64
        self.head = nn.Conv2d(b, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)  # raw logits


class SMPWrapper(nn.Module):
    """Wraps an smp model; ImageNet normalisation inside forward."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, pretrained=False):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch not installed; "
                "pip install segmentation-models-pytorch (only needed for --arch smp_unet)"
            ) from e
        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build_model(arch, pretrained=False):
    """arch: 'unet' (plain, from scratch) or 'smp_unet' (resnet34 encoder)."""
    if arch == "unet":
        return UNet()
    if arch == "smp_unet":
        return SMPWrapper(pretrained=pretrained)
    raise ValueError(f"unknown arch '{arch}', expected 'unet' or 'smp_unet'")


def count_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = UNet()
    print("UNet params:", count_params(m))
    y = m(torch.randn(1, 3, 512, 512))
    print("output shape (1,3,512,512) ->", tuple(y.shape))
