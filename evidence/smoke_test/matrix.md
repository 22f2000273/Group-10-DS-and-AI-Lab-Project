# Smoke test only - 2 epochs, 8 tiles per split. Values prove the pipeline runs; they have no scientific meaning.

# Cross-dataset matrices (arch=unet)

## IoU

| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.0015 | 0.0039 | 0.0000 |
| inria | 0.0528 | 0.1828 | 0.1490 |
| mass | 0.0292 | 0.1367 | 0.1464 |

## RTD %

| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.0000 | -160.97 | 98.70 |
| inria | 71.11 | 0.0000 | 18.48 |
| mass | 80.08 | 6.62 | 0.0000 |

