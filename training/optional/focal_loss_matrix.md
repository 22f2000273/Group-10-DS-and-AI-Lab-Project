# ALTERNATIVE LOSS COMPARISON (focal + Dice, FIXED) - comparison only, not part of the main baseline matrix.

Rebuilt after the FocalDice bug fix (Dice eps=1.0 leaked into the focal log
terms). The old FOCAL_* runs are INVALID; numbers below come from the
FOCALFIX_* re-runs (Protocol A, seed 42).

# Diagonal IoU: baseline (BCE+Dice) vs focal+Dice (FIXED), per training dataset
| train_set | baseline IoU | focal IoU (fixed) | buggy focal (invalid) | difference vs baseline |
|---|---|---|---|---|
| whu | 0.8624 | 0.8509 | 0.6314 | -0.0115 |
| inria | 0.7169 | 0.6859 | 0.5633 | -0.0310 |
| mass | 0.6665 | 0.6683 | 0.5279 | +0.0018 |

Verdict: with the fix, focal+Dice is competitive with the baseline on all
three diagonals (worst gap -0.031 on Inria, +0.002 on Mass). The old
"loss collapse" conclusion was an artifact of the bug. Focal still does not
beat the baseline consistently, so BCE+Dice stays the default.

# Cross-dataset matrices (arch=unet)

## IoU

| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.8509 | 0.3262 | 0.3759 |
| inria | 0.7102 | 0.6859 | 0.3035 |
| mass | 0.5789 | 0.1857 | 0.6683 |

## RTD %

| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.0000 | 61.67 | 55.82 |
| inria | -3.54 | 0.0000 | 55.76 |
| mass | 13.38 | 72.21 | 0.0000 |
