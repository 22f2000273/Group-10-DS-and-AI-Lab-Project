# Milestone 4 — Optional Experiments FINAL REPORT

## Phase 0 — Setup verification (no training)
- git pull: up to date; latest commit `3c3d947` ("Add --max_steps flag and smoke-test evidence")
- pytest: **12 passed** ✅
- `--loss` flag in train.py: did NOT exist; added `--loss {bce_dice,focal_dice}` (default `bce_dice`)
- segmentation-models-pytorch: NOT installed; installed **smp 0.6.0** via pip
- Disk: 3.1 TB free · GPU: NVIDIA L40S, 45 GB free

## Phase 1 — ResNet-34 pre-trained variant (VARIANT — comparison only)
- Dry-run test (`--arch smp_unet --pretrained --dry_run` on WHU): PASSED — full flow complete, **24,436,369 params** (vs baseline U-Net 31,037,633)
- Runs:

| run | best epoch | best val IoU | test whu | test inria | test mass | wall time |
|---|---|---|---|---|---|---|
| variant_mass_seed42 | 42/50 | 0.6620 | 0.5960 | 0.2169 | **0.6808** | ~6 min |
| variant_whu_seed42 | 49/50 | 0.8951 | **0.8593** | 0.4229 | 0.3610 | ~40 min |
| variant_inria_seed42 | 49/50 | 0.7400 | 0.6656 | **0.7224** | 0.3246 | ~40 min |

- Variant diagonal vs baseline diagonal: Mass 0.6808 vs 0.6665 (+), WHU 0.8593 vs 0.8624 (=), Inria 0.7224 vs 0.7169 (+)
- Matrices: `training/optional/resnet_variant_matrix.md` (+ heading: "OPTIONAL VARIANT (ResNet-34 pre-trained encoder) — comparison only, not part of the main baseline matrix.")

## Phase 2 — Alternative loss: focal + Dice
- `--loss {bce_dice,focal_dice}` added to train.py (default `bce_dice`, existing behavior unchanged); `FocalDice` class (gamma 2.0 + same Dice term); only `src/evaluation/train.py` changed; pytest **12 passed**; `--dry_run --loss focal_dice` on WHU completed with finite loss

| run | best epoch | best val IoU | test whu | test inria | test mass | wall time |
|---|---|---|---|---|---|---|
| focal_mass_seed42 | 11/21* | 0.5698 | 0.4266 | 0.1563 | **0.5279** | ~8 min |
| focal_whu_seed42 | 2/12* | 0.6500 | **0.6314** | 0.1281 | 0.1699 | ~35 min |
| focal_inria_seed42 | 48/50 | 0.5780 | 0.4725 | **0.5633** | 0.2738 | ~5.5 h |

*\*early-stopped by patience 10.*

- Diagonal IoU: baseline (BCE+Dice) vs focal+Dice — whu 0.8624 → **0.6314** (−0.2310), inria 0.7169 → **0.5633** (−0.1536), mass 0.6665 → **0.5279** (−0.1386)
- Focal+Dice is **worse than BCE+Dice on every test set** in every run.
- Matrices: `training/optional/focal_loss_matrix.md` (includes diagonal comparison table) + csv.

## Phase 3 — Inference latency (measurement only; no training)
- GPU: NVIDIA L40S · batch size: 1 · tile size: 512×512 · 10 warm-up tiles
- Mean ms per tile over up to 100 timed test tiles (Mass timed 30 — its test split has 40 tiles):

| dataset | tiles timed | mean ms/tile | std ms/tile | tiles/sec |
|---|---|---|---|---|
| whu | 100 | 7.90 | 0.02 | 126.7 |
| inria | 100 | 7.91 | 0.01 | 126.5 |
| mass | 30 | 7.93 | 0.02 | 126.1 |

- Saved: `training/optional/latency.md` + `training/optional/latency.py`

## Incidents / failures
- None. All 6 optional runs completed; Mass variant took ~6 min, WHU variant ~40 min, Inria variant ~40 min; focal runs ~8 / 35 / 330 min. Validation loss printed as `nan` for several focal epochs on Inria — training continued normally.

## Files under training/optional/ ready to commit
- PROGRESS.md, optional_tracker.csv
- summaries/VARIANT_*.txt (3), summaries/FOCAL_*.txt (3)
- logs/VARIANT_*.log (3), logs/FOCAL_*.log (3)
- resnet_variant_matrix.md, resnet_variant_matrix_iou.csv, resnet_variant_matrix_rtd.csv
- focal_loss_matrix.md, focal_loss_matrix_iou.csv, focal_loss_matrix_rtd.csv
- latency.md, latency.py
- run folders remain in runs/optional/ (checkpoints *.pt — gitignored, stay on server per standing rules)
