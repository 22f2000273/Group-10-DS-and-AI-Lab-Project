# OPTIONAL VARIANT (ResNet-34 pre-trained encoder) — comparison only, not part of the main baseline matrix.

# Cross-dataset matrices (arch=smp_unet)

## IoU

| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.8593 | 0.4229 | 0.3610 |
| inria | 0.6656 | 0.7224 | 0.3246 |
| mass | 0.5960 | 0.2169 | 0.6808 |

## RTD %

| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.0000 | 50.79 | 57.99 |
| inria | 7.87 | 0.0000 | 55.07 |
| mass | 12.45 | 68.14 | 0.0000 |
