# ALTERNATIVE LOSS COMPARISON (focal + Dice) — comparison only, not part of the main baseline matrix.

# Diagonal IoU: baseline (BCE+Dice) vs focal+Dice, per training dataset
| train_set | baseline IoU | focal IoU | difference |
|---|---|---|---|
| whu | 0.8624 | 0.6314 | -0.2310 |
| inria | 0.7169 | 0.5633 | -0.1536 |
| mass | 0.6665 | 0.5279 | -0.1386 |

# Cross-dataset matrices (arch=unet)

## IoU

| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.6314 | 0.1281 | 0.1699 |
| inria | 0.4725 | 0.5633 | 0.2738 |
| mass | 0.4266 | 0.1563 | 0.5279 |

## RTD %

| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.0000 | 79.72 | 73.10 |
| inria | 16.11 | 0.0000 | 51.39 |
| mass | 19.19 | 70.40 | 0.0000 |
